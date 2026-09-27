"""Pydantic models for JWT claims and the WebSocket wire protocol."""

from typing import Literal

from pydantic import BaseModel, Field

Department = Literal["hr", "finance", "exec"]


class UserClaims(BaseModel):
    """Identity taken from the verified JWT. Never from a chat message."""

    sub: str
    email: str | None = None
    department: Department
    level: int = Field(ge=1, le=3)


# --- client -> server -------------------------------------------------------

class AuthRequest(BaseModel):
    type: Literal["auth"]
    token: str


class ChatRequest(BaseModel):
    type: Literal["message"]
    text: str = Field(min_length=1, max_length=4000)


# --- server -> client -------------------------------------------------------

class AuthSuccess(BaseModel):
    type: Literal["auth_success"] = "auth_success"
    user_id: str
    department: Department
    level: int


class AuthFailed(BaseModel):
    type: Literal["auth_failed"] = "auth_failed"
    message: str


class StreamChunk(BaseModel):
    type: Literal["stream"] = "stream"
    text: str


class Done(BaseModel):
    type: Literal["done"] = "done"


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    message: str
