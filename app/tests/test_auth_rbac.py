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


def test_admin_can_list_users(client, admin_auth):
    r = client.get("/auth/users", headers=admin_auth)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert len(body) >= 1


def test_non_admin_cannot_list_users(client, user_auth):
    r = client.get("/auth/users", headers=user_auth)
    assert r.status_code == 403


def test_admin_can_create_user_with_role(client, admin_auth):
    r = client.post(
        "/auth/users",
        headers=admin_auth,
        json={"email": "created_recruiter@example.com", "password": "Test123!", "role": "recruiter"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["email"] == "created_recruiter@example.com"
    assert body["role"] == "recruiter"


def test_admin_can_create_admin_user(client, admin_auth):
    r = client.post(
        "/auth/users",
        headers=admin_auth,
        json={"email": "created_admin@example.com", "password": "Test123!", "role": "admin"},
    )
    assert r.status_code == 201
    assert r.json()["role"] == "admin"


def test_non_admin_cannot_create_user(client, user_auth):
    r = client.post(
        "/auth/users",
        headers=user_auth,
        json={"email": "blocked@example.com", "password": "Test123!", "role": "user"},
    )
    assert r.status_code == 403


def test_admin_can_delete_user(client, admin_auth):
    created = client.post(
        "/auth/users",
        headers=admin_auth,
        json={"email": "delete_me@example.com", "password": "Test123!", "role": "user"},
    )
    assert created.status_code == 201
    user_id = created.json()["id"]

    deleted = client.delete(f"/auth/users/{user_id}", headers=admin_auth)
    assert deleted.status_code == 204

    listed = client.get("/auth/users", headers=admin_auth)
    assert listed.status_code == 200
    assert all(row["id"] != user_id for row in listed.json())


def test_admin_cannot_delete_self(client, admin_auth):
    listed = client.get("/auth/users", headers=admin_auth)
    assert listed.status_code == 200
    admin_id = listed.json()[0]["id"]

    deleted = client.delete(f"/auth/users/{admin_id}", headers=admin_auth)
    assert deleted.status_code == 400
    assert deleted.json()["detail"]["code"] == "self_user_delete_forbidden"


def test_non_admin_cannot_delete_user(client, admin_auth, user_auth):
    created = client.post(
        "/auth/users",
        headers=admin_auth,
        json={"email": "delete_blocked@example.com", "password": "Test123!", "role": "user"},
    )
    assert created.status_code == 201

    deleted = client.delete(f"/auth/users/{created.json()['id']}", headers=user_auth)
    assert deleted.status_code == 403
