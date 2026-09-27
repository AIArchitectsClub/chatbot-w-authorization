"""The /ws/chat WebSocket endpoint.

Protocol:  client -> {"type":"auth","token":...}
           server -> {"type":"auth_success",...} | {"type":"auth_failed",...} + close
           client -> {"type":"message","text":...}   (repeatable)
           server -> {"type":"stream","text":...}*, then {"type":"done"}
           On a per-message failure the server sends {"type":"error",...} then
           {"type":"done"} and keeps the connection open.
"""

import asyncio
import logging

from agno.agent import Agent
from agno.run.agent import RunContentEvent, RunErrorEvent
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from app.auth import AuthError, verify_token
from app.config import Settings
from app.llm.agent import build_agent
from app.schemas import (
    AuthFailed,
    AuthRequest,
    AuthSuccess,
    ChatRequest,
    Done,
    ErrorMessage,
    StreamChunk,
    UserClaims,
)

logger = logging.getLogger(__name__)
router = APIRouter()

WS_POLICY_VIOLATION = 1008
GENERIC_ERROR = "Sorry, something went wrong while answering. Please try again."


async def send(ws: WebSocket, message: BaseModel) -> None:
    await ws.send_text(message.model_dump_json())


async def authenticate(ws: WebSocket, settings: Settings) -> UserClaims | None:
    """Read the first message and verify its JWT. On failure, reply and close."""
    try:
        raw = await asyncio.wait_for(ws.receive_text(), settings.auth_timeout_seconds)
        request = AuthRequest.model_validate_json(raw)
        return verify_token(request.token, settings)
    except asyncio.TimeoutError:
        reason = "Authentication timed out"
    except ValidationError:
        reason = 'First message must be {"type": "auth", "token": "<jwt>"}'
    except AuthError as exc:
        reason = str(exc)

    await send(ws, AuthFailed(message=reason))
    await ws.close(code=WS_POLICY_VIOLATION)
    return None


async def stream_answer(ws: WebSocket, agent: Agent, text: str) -> None:
    """Forward the agent's tokens to the client as they are produced."""
    try:
        async for event in agent.arun(text, stream=True):
            if isinstance(event, RunErrorEvent):
                raise RuntimeError(event.content)
            if isinstance(event, RunContentEvent) and isinstance(event.content, str):
                await send(ws, StreamChunk(text=event.content))
    except WebSocketDisconnect:
        raise
    except Exception:
        # LLM / tool / retrieval failure: tell the user, keep the connection open.
        logger.exception("failed to answer message")
        await send(ws, ErrorMessage(message=GENERIC_ERROR))
    await send(ws, Done())


async def handle_message(ws: WebSocket, agent: Agent, raw: str) -> None:
    try:
        request = ChatRequest.model_validate_json(raw)
    except ValidationError:
        await send(ws, ErrorMessage(message='Expected {"type": "message", "text": "..."}'))
        return
    await stream_answer(ws, agent, request.text)


@router.websocket("/ws/chat")
async def chat(ws: WebSocket) -> None:
    await ws.accept()
    settings: Settings = ws.app.state.settings

    user = await authenticate(ws, settings)
    if user is None:
        return

    agent = build_agent(user, ws.app.state.vector_db, settings)
    await send(ws, AuthSuccess(user_id=user.sub, department=user.department, level=user.level))
    logger.info("connected user=%s dept=%s level=%d", user.sub, user.department, user.level)

    try:
        while True:
            await handle_message(ws, agent, await ws.receive_text())
    except WebSocketDisconnect:
        logger.info("disconnected user=%s", user.sub)
