"""JWT verification."""

import jwt
from pydantic import ValidationError

from app.config import Settings
from app.schemas import UserClaims


class AuthError(Exception):
    """Raised when a token is missing, invalid, expired or malformed."""


def verify_token(token: str, settings: Settings) -> UserClaims:
    """Verify signature and expiry, and return the caller's claims."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],  # pinned: never trust the token's own alg
            options={"require": ["exp", "sub"]},
        )
        return UserClaims.model_validate(payload)
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("Invalid token") from exc
    except ValidationError as exc:
        raise AuthError("Token is missing required claims") from exc
