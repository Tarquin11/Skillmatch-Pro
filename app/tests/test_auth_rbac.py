def test_auth_signup_login_me(client):
    email = "rbac_user@example.com"
    password = "Test123!"

    r = client.post("/auth/signup", json={"email": email, "password": password, "role": "user"})
    assert r.status_code == 200

    r = client.post(
        "/auth/login",
        data={"username": email, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 200
    token = r.json()["access_token"]

    r = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["email"] == email


def test_rbac_user_cannot_create_skill(client, user_auth):
    r = client.post("/skills/", headers=user_auth, json={"name": "PyTest Skill"})
    assert r.status_code == 403


def test_rbac_admin_can_create_skill(client, admin_auth):
    r = client.post("/skills/", headers=admin_auth, json={"name": "PyTest Skill"})
    assert r.status_code == 201
