import pytest

from app.auth import AuthError, verify_token


def test_valid_token_yields_claims(settings, make_token):
    claims = verify_token(make_token(sub="mgr-002", level=2), settings)
    assert (claims.sub, claims.department, claims.level) == ("mgr-002", "hr", 2)


def test_expired_token_rejected(settings, make_token):
    with pytest.raises(AuthError, match="expired"):
        verify_token(make_token(ttl=-10), settings)


def test_wrong_signature_rejected(settings, make_token):
    with pytest.raises(AuthError):
        verify_token(make_token(secret="another-secret-another-secret-123"), settings)


def test_garbage_token_rejected(settings):
    with pytest.raises(AuthError):
        verify_token("not-a-jwt", settings)


def test_missing_claims_rejected(settings):
    import jwt

    token = jwt.encode({"sub": "x", "exp": 9999999999}, settings.jwt_secret, algorithm="HS256")
    with pytest.raises(AuthError, match="claims"):
        verify_token(token, settings)


def test_unknown_department_rejected(settings, make_token):
    with pytest.raises(AuthError):
        verify_token(make_token(department="marketing"), settings)


def test_alg_none_token_rejected(settings):
    import jwt

    token = jwt.encode({"sub": "x", "department": "hr", "level": 3, "exp": 9999999999}, "", algorithm="none")
    with pytest.raises(AuthError):
        verify_token(token, settings)
