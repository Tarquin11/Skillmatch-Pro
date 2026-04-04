import uuid
from datetime import timedelta

from app.core.security import create_access_token


def _signup(client, *, role: str = "user") -> tuple[str, str]:
    email = f"sec_{uuid.uuid4().hex[:8]}@example.com"
    password = "Test123!"
    r = client.post("/auth/signup", json={"email": email, "password": password, "role": role})
    assert r.status_code == 200, r.text
    return email, password


def _login(client, email: str, password: str) -> dict:
    r = client.post(
        "/auth/login",
        data={"username": email, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_forbidden_access_without_token(client):
    r = client.get("/skills/")
    assert r.status_code == 401


def test_invalid_and_expired_tokens_are_rejected(client):
    email, _ = _signup(client, role="user")

    invalid = client.get("/auth/me", headers={"Authorization": "Bearer not-a-jwt"})
    assert invalid.status_code == 401, invalid.text

    expired_token = create_access_token(
        data={"sub": email},
        expires_delta=timedelta(minutes=-1),
        token_version=0,
    )
    expired = client.get("/auth/me", headers={"Authorization": f"Bearer {expired_token}"})
    assert expired.status_code == 401, expired.text


def test_refresh_token_reuse_revokes_session_tokens(client):
    email, password = _signup(client, role="user")
    login = _login(client, email, password)
    access_1 = login["access_token"]
    refresh_1 = login["refresh_token"]

    rotate = client.post("/auth/refresh", json={"refresh_token": refresh_1})
    assert rotate.status_code == 200, rotate.text
    access_2 = rotate.json()["access_token"]

    reuse = client.post("/auth/refresh", json={"refresh_token": refresh_1})
    assert reuse.status_code == 401, reuse.text
    detail = reuse.json().get("detail", {})
    assert detail.get("code") == "refresh_token_reuse_detected"

    stale_1 = client.get("/auth/me", headers={"Authorization": f"Bearer {access_1}"})
    stale_2 = client.get("/auth/me", headers={"Authorization": f"Bearer {access_2}"})
    assert stale_1.status_code == 401, stale_1.text
    assert stale_2.status_code == 401, stale_2.text


def test_non_bootstrap_user_cannot_self_assign_admin_role(client, admin_auth):
    # First admin user already exists via fixture; a second signup cannot request admin.
    email = f"sec_admin_{uuid.uuid4().hex[:8]}@example.com"
    payload = {"email": email, "password": "Test123!", "role": "admin"}
    r = client.post("/auth/signup", json=payload)
    assert r.status_code == 403, r.text
    detail = r.json().get("detail", {})
    assert detail.get("code") == "admin_signup_forbidden"

