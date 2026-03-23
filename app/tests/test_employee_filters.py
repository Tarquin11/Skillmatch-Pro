import uuid


def _create_employee(client, auth, dep, pos, suffix):
    payload = {
        "employeeNumber": f"EMP-{suffix}",
        "first_name": "Test",
        "last_name": suffix,
        "full_name": f"Test {suffix}",
        "email": f"filter_{suffix}@example.com",
        "departement": dep,
        "position": pos,
    }
    r = client.post("/employees/", headers=auth, json=payload)
    assert r.status_code == 201, r.text


def test_filter_department_and_position_regression(client, admin_auth):
    s1 = uuid.uuid4().hex[:6]
    s2 = uuid.uuid4().hex[:6]
    s3 = uuid.uuid4().hex[:6]

    _create_employee(client, admin_auth, "IT", "IT Support Specialist", s1)
    _create_employee(client, admin_auth, "IT", "Software Developer", s2)
    _create_employee(client, admin_auth, "Sales", "Support Specialist", s3)

    r = client.get(
        "/employees/",
        headers=admin_auth,
        params={"department": "IT", "position": "support", "limit": 50},
    )
    assert r.status_code == 200, r.text
    rows = r.json()

    assert len(rows) >= 1
    assert all("IT" in (row.get("department") or "") for row in rows)
    assert all("support" in (row.get("position") or "").lower() for row in rows)
