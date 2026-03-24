import uuid
import pytest
from fastapi.testclient import TestClient


def _headers_for_role(role: str, admin_auth: dict[str, str], user_auth: dict[str, str]) -> dict[str, str]:
    if role == "admin":
        return admin_auth
    if role == "user":
        return user_auth
    return {}


def _request(
    client: TestClient,
    method: str,
    path: str,
    headers: dict[str, str],
    json_payload: dict | None = None,
):
    kwargs = {"headers": headers} if headers else {}
    if json_payload is not None:
        kwargs["json"] = json_payload
    return client.request(method=method, url=path, **kwargs)


def _create_skill(client: TestClient, admin_auth: dict[str, str], name: str) -> int:
    r = client.post("/skills/", headers=admin_auth, json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _create_employee(client: TestClient, admin_auth: dict[str, str], suffix: str) -> int:
    payload = {
        "employeeNumber": f"EMP-{suffix}",
        "first_name": "Matrix",
        "last_name": "User",
        "full_name": "Matrix User",
        "email": f"matrix.{suffix}@example.com",
        "department": "Engineering",
        "position": "Engineer",
    }
    r = client.post("/employees/", headers=admin_auth, json=payload)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _create_job(client: TestClient, admin_auth: dict[str, str], suffix: str) -> int:
    payload = {
        "title": f"Matrix Job {suffix}",
        "description": "Permission matrix seed job",
        "departement": "Engineering",
    }
    r = client.post("/jobs/", headers=admin_auth, json=payload)
    assert r.status_code == 201, r.text
    return r.json()["id"]


@pytest.fixture()
def matrix_seed(client, admin_auth):
    suffix = uuid.uuid4().hex[:8]
    skill_id = _create_skill(client, admin_auth, f"MatrixSeedSkill-{suffix}")
    employee_id = _create_employee(client, admin_auth, suffix)
    job_id = _create_job(client, admin_auth, suffix)
    return {
        "nonce": suffix,
        "skill_id": skill_id,
        "employee_id": employee_id,
        "job_id": job_id,
    }


CASES = [
    {
        "id": "auth_me_get",
        "method": "GET",
        "path": "/auth/me",
        "expected": {"admin": 200, "user": 200, "anon": 401},
    },
    {
        "id": "skills_list_get",
        "method": "GET",
        "path": "/skills/",
        "expected": {"admin": 200, "user": 200, "anon": 401},
    },
    {
        "id": "skills_create_post",
        "method": "POST",
        "path": "/skills/",
        "json": lambda c: {"name": f"MatrixSkill-{c['nonce']}"},
        "expected": {"admin": 201, "user": 403, "anon": 401},
    },
    {
        "id": "skills_delete_delete",
        "method": "DELETE",
        "path": lambda c: f"/skills/{c['skill_id']}",
        "expected": {"admin": 204, "user": 403, "anon": 401},
    },
    {
        "id": "departments_create_post",
        "method": "POST",
        "path": "/departments/",
        "json": lambda c: {"name": f"Matrix Department {c['nonce']}"},
        "expected": {"admin": 201, "user": 403, "anon": 401},
    },
    {
        "id": "jobs_create_post",
        "method": "POST",
        "path": "/jobs/",
        "json": lambda c: {"title": f"Matrix New Job {c['nonce']}"},
        "expected": {"admin": 201, "user": 403, "anon": 401},
    },
    {
        "id": "employees_create_post",
        "method": "POST",
        "path": "/employees/",
        "json": lambda c: {
            "employeeNumber": f"EMP-NEW-{c['nonce']}",
            "first_name": "Role",
            "last_name": "Matrix",
            "full_name": "Role Matrix",
            "email": f"new.{c['nonce']}@example.com",
        },
        "expected": {"admin": 201, "user": 403, "anon": 401},
    },
    {
        "id": "job_skill_upsert_post",
        "method": "POST",
        "path": lambda c: f"/jobs/{c['job_id']}/skills",
        "json": lambda c: {"skill_id": c["skill_id"], "required_level": 3, "weight": 1.0},
        "expected": {"admin": 200, "user": 403, "anon": 401},
    },
    {
        "id": "employee_skill_assign_post",
        "method": "POST",
        "path": lambda c: f"/skills/employees/{c['employee_id']}",
        "json": lambda c: {"skill_id": c["skill_id"], "level": 3},
        "expected": {"admin": 200, "user": 403, "anon": 401},
    },
    {
        "id": "legacy_departements_get",
        "method": "GET",
        "path": "/departements/",
        "expected": {"admin": 200, "user": 200, "anon": 401},
    },
]


@pytest.mark.parametrize("role", ["admin", "user", "anon"])
@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_permission_matrix(client, admin_auth, user_auth, matrix_seed, role, case):
    headers = _headers_for_role(role, admin_auth, user_auth)

    path = case["path"](matrix_seed) if callable(case["path"]) else case["path"]
    payload_builder = case.get("json")
    payload = payload_builder(matrix_seed) if callable(payload_builder) else payload_builder

    response = _request(
        client=client,
        method=case["method"],
        path=path,
        headers=headers,
        json_payload=payload,
    )

    expected = case["expected"][role]
    assert response.status_code == expected, (
        f"case={case['id']} role={role} expected={expected} "
        f"got={response.status_code} body={response.text}"
    )
