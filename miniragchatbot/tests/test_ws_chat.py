"""WebSocket protocol tests, with the agent and vector store faked out."""

import json

import pytest
from agno.run.agent import RunContentEvent
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app import main as main_module
from app.api import ws_chat


class FakeAgent:
    """Yields two tokens, or raises for the text 'boom'."""

    def __init__(self, user):
        self.user = user

    async def arun(self, text, stream=True):
        if text == "boom":
            raise RuntimeError("llm down")
        for token in ("You are entitled to ", "22 working days"):
            yield RunContentEvent(content=token)


@pytest.fixture
def client(settings, monkeypatch):
    monkeypatch.setattr(main_module, "make_embedder", lambda s: None)
    monkeypatch.setattr(main_module, "make_vector_db", _FakeVectorDb)
    monkeypatch.setattr(ws_chat, "build_agent", lambda user, db, s: FakeAgent(user))
    with TestClient(main_module.create_app(settings)) as c:
        yield c


class _FakeVectorDb:
    def __init__(self, *_):
        self.async_client = self

    async def close(self):
        pass


def recv(ws) -> dict:
    return json.loads(ws.receive_text())


def auth(ws, token):
    ws.send_text(json.dumps({"type": "auth", "token": token}))
    return recv(ws)


def test_happy_path_streams_tokens_then_done(client, make_token):
    with client.websocket_connect("/ws/chat") as ws:
        assert auth(ws, make_token(sub="mgr-002", level=2)) == {
            "type": "auth_success", "user_id": "mgr-002", "department": "hr", "level": 2,
        }
        ws.send_text(json.dumps({"type": "message", "text": "How many leave days?"}))
        assert recv(ws) == {"type": "stream", "text": "You are entitled to "}
        assert recv(ws) == {"type": "stream", "text": "22 working days"}
        assert recv(ws) == {"type": "done"}


@pytest.mark.parametrize("token", ["garbage", ""])
def test_bad_token_gets_auth_failed_and_socket_closes(client, token):
    with client.websocket_connect("/ws/chat") as ws:
        reply = auth(ws, token)
        assert reply["type"] == "auth_failed"
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()


def test_expired_token_rejected(client, make_token):
    with client.websocket_connect("/ws/chat") as ws:
        assert auth(ws, make_token(ttl=-5)) == {"type": "auth_failed", "message": "Token has expired"}


def test_non_auth_first_message_rejected(client):
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({"type": "message", "text": "hi"}))
        assert recv(ws)["type"] == "auth_failed"


def test_auth_timeout(client):
    with client.websocket_connect("/ws/chat") as ws:  # settings fixture: 1s timeout
        assert recv(ws) == {"type": "auth_failed", "message": "Authentication timed out"}


def test_llm_failure_sends_error_and_keeps_connection_open(client, make_token):
    with client.websocket_connect("/ws/chat") as ws:
        auth(ws, make_token())
        ws.send_text(json.dumps({"type": "message", "text": "boom"}))
        assert recv(ws)["type"] == "error"
        assert recv(ws) == {"type": "done"}

        # connection still usable
        ws.send_text(json.dumps({"type": "message", "text": "again"}))
        assert recv(ws)["type"] == "stream"


def test_invalid_message_shape_is_reported_not_fatal(client, make_token):
    with client.websocket_connect("/ws/chat") as ws:
        auth(ws, make_token())
        ws.send_text("not json")
        assert recv(ws)["type"] == "error"
        ws.send_text(json.dumps({"type": "message", "text": "ok"}))
        assert recv(ws)["type"] == "stream"
