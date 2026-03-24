import uuid


def _create_skill(client, admin_auth, name: str) -> int:
    r = client.post("/skills/", headers=admin_auth, json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_update_skill_requires_if_match(client, admin_auth):
    sid = _create_skill(client, admin_auth, f"ConcurrencySkill-{uuid.uuid4().hex[:6]}")

    # Missing If-Match -> 428
    r = client.put(f"/skills/{sid}", headers=admin_auth, json={"name": "Renamed Skill"})
    assert r.status_code == 428, r.text
    body = r.json()
    assert body["detail"]["code"] == "precondition_required"


def test_update_skill_rejects_stale_etag(client, admin_auth):
    sid = _create_skill(client, admin_auth, f"ConcurrencySkill-{uuid.uuid4().hex[:6]}")

    # Get current ETag
    r = client.get(f"/skills/{sid}", headers=admin_auth)
    assert r.status_code == 200, r.text
    etag = r.headers.get("ETag")
    assert etag

    # First update with valid ETag -> 200
    r = client.put(
        f"/skills/{sid}",
        headers={**admin_auth, "If-Match": etag},
        json={"name": "Skill v2"},
    )
    assert r.status_code == 200, r.text
    new_etag = r.headers.get("ETag")
    assert new_etag and new_etag != etag

    # Reuse old stale ETag -> 412
    r = client.put(
        f"/skills/{sid}",
        headers={**admin_auth, "If-Match": etag},
        json={"name": "Skill v3"},
    )
    assert r.status_code == 412, r.text
    body = r.json()
    assert body["detail"]["code"] == "etag_mismatch"


def test_update_skill_accepts_wildcard_if_match(client, admin_auth):
    sid = _create_skill(client, admin_auth, f"ConcurrencySkill-{uuid.uuid4().hex[:6]}")

    r = client.put(
        f"/skills/{sid}",
        headers={**admin_auth, "If-Match": "*"},
        json={"name": "Wildcard Update"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "Wildcard Update"
