import os
import time

import jwt
import pytest

# Must be set before app modules read settings.
os.environ.setdefault("JWT_SECRET", "test-secret-test-secret-test-secret")

from app.config import Settings  # noqa: E402


@pytest.fixture
def settings() -> Settings:
    return Settings(jwt_secret=os.environ["JWT_SECRET"], auth_timeout_seconds=1.0)


@pytest.fixture
def make_token(settings: Settings):
    def _make(sub="emp-001", department="hr", level=1, ttl=3600, secret=None, **extra) -> str:
        now = int(time.time())
        payload = {"sub": sub, "department": department, "level": level, "iat": now, "exp": now + ttl}
        return jwt.encode({**payload, **extra}, secret or settings.jwt_secret, algorithm="HS256")

    return _make
