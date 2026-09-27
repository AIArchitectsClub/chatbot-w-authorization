"""Tiny command-line WebSocket client for trying the chatbot without a browser.

Usage (from the project root, server running):
    python -m scripts.ws_demo emp-001 "How many leave days do I get?"
    python -m scripts.ws_demo exec-003 "What are the APAC priorities?" "Who is my manager?"
    python -m scripts.ws_demo bad-token "hello"        # shows auth_failed

Tokens are minted locally with the same JWT_SECRET as the server.
"""

import asyncio
import json
import sys
import time

import jwt
import websockets

from app.config import get_settings

USERS = {
    "emp-001": {"email": "emp@test.com", "department": "hr", "level": 1},
    "mgr-002": {"email": "mgr@test.com", "department": "hr", "level": 2},
    "exec-003": {"email": "exec@test.com", "department": "exec", "level": 3},
}
URL = "ws://localhost:8000/ws/chat"


def mint(user_id: str) -> str:
    settings = get_settings()
    if user_id not in USERS:
        return user_id  # treat the argument as a raw (probably invalid) token
    now = int(time.time())
    payload = {"sub": user_id, **USERS[user_id], "iat": now, "exp": now + 3600}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


async def main(user: str, questions: list[str]) -> None:
    async with websockets.connect(URL) as ws:
        await ws.send(json.dumps({"type": "auth", "token": mint(user)}))
        reply = json.loads(await ws.recv())
        print(f"<- {reply}")
        if reply["type"] != "auth_success":
            return
        for question in questions:
            print(f"\n-> {question}")
            await ws.send(json.dumps({"type": "message", "text": question}))
            started = time.perf_counter()
            first: float | None = None
            while True:
                event = json.loads(await ws.recv())
                if event["type"] == "stream":
                    first = first or time.perf_counter() - started
                    print(event["text"], end="", flush=True)
                elif event["type"] == "done":
                    print(f"\n   [done in {time.perf_counter() - started:.1f}s, first token {first or 0:.1f}s]")
                    break
                else:
                    print(f"\n<- {event}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    asyncio.run(main(sys.argv[1], sys.argv[2:]))
