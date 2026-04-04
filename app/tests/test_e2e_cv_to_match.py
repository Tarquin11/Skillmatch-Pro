import uuid

from app.api import candidates as candidates_api
from app.api import match as match_api
from app.schemas.match import JobMatchResponse


def _create_skill(client, admin_auth, name: str) -> int:
    r = client.post("/skills/", headers=admin_auth, json={"name": name})
    assert r.status_code == 201, r.text
    return int(r.json()["id"])


def _create_employee(client, admin_auth, *, position: str) -> int:
    uid = uuid.uuid4().hex[:8]
    payload = {
        "employeeNumber": f"EMP-{uid}",
        "first_name": "E2E",
        "last_name": "Candidate",
        "full_name": "E2E Candidate",
        "email": f"e2e_{uid}@example.com",
        "departement": "IT",
        "position": position,
    }
    r = client.post("/employees/", headers=admin_auth, json=payload)
    assert r.status_code == 201, r.text
    return int(r.json()["id"])


def _assign_skill_to_employee(client, admin_auth, employee_id: int, skill_id: int, level: int = 4) -> None:
    r = client.post(
        f"/skills/employees/{employee_id}",
        headers=admin_auth,
        json={"skill_id": skill_id, "level": level},
    )
    assert r.status_code == 200, r.text


def test_e2e_cv_upload_parse_create_candidate_then_match(client, admin_auth, monkeypatch):
    monkeypatch.setattr(match_api.inference_service, "_DRIFT_MONITORING_ENABLED", False, raising=False)
    monkeypatch.setattr(match_api.inference_service, "_select_model_matcher", lambda **_: (None, "heuristic"), raising=False)

    fake_parsed = {
        "ok": True,
        "degraded": False,
        "errors": [],
        "warnings": [],
        "text_length": 240,
        "skills": ["python", "sql", "docker"],
        "extracted_skills": [
            {"skill": "python", "confidence": 0.99, "source": "lexicon"},
            {"skill": "sql", "confidence": 0.98, "source": "lexicon"},
            {"skill": "docker", "confidence": 0.97, "source": "lexicon"},
        ],
        "preview": "Backend engineer profile",
        "predicted_title": "Backend Engineer",
        "predicted_experience_years": 2.0,
    }
    monkeypatch.setattr(candidates_api, "parse_cv_safe", lambda **_: fake_parsed, raising=False)

    skill_ids = {
        "python": _create_skill(client, admin_auth, "python"),
        "sql": _create_skill(client, admin_auth, "sql"),
        "docker": _create_skill(client, admin_auth, "docker"),
    }

    files = {"file": ("candidate.pdf", b"%PDF-1.7\n%fake cv content", "application/pdf")}
    upload = client.post("/candidates/upload_cv", headers=admin_auth, files=files)
    assert upload.status_code == 200, upload.text
    upload_body = upload.json()
    assert upload_body["ok"] is True
    assert upload_body["degraded"] is False
    assert set(upload_body["skills"]) >= {"python", "sql", "docker"}
    assert upload_body["predicted_title"] == "Backend Engineer"

    employee_id = _create_employee(client, admin_auth, position=upload_body["predicted_title"])
    for skill_name in upload_body["skills"]:
        if skill_name in skill_ids:
            _assign_skill_to_employee(client, admin_auth, employee_id, skill_ids[skill_name], level=4)

    payload = {
        "job_title": "Backend Engineer",
        "required_skills": ["python", "sql", "docker"],
        "min_experience": 0,
        "limit": 10,
    }
    match_response = client.post("/match/job", headers=admin_auth, json=payload)
    assert match_response.status_code == 200, match_response.text
    body = match_response.json()
    parsed = JobMatchResponse(**body)

    assert len(parsed.results) >= 1
    assert len(parsed.results) <= payload["limit"]
    assert parsed.results[0].employee_id == employee_id
    assert all(
        parsed.results[i].predicted_fit_score >= parsed.results[i + 1].predicted_fit_score
        for i in range(len(parsed.results) - 1)
    )
    assert parsed.results[0].scoring_source
