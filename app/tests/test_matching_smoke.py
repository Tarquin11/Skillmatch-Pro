import uuid
from app.api import candidates as candidates_api

def test_match_job_smoke(client, admin_auth):
    uid = uuid.uuid4().hex[:6]
    payload = {
        "employeeNumber": f"EMP-{uid}",
        "first_name": "AI",
        "last_name": "Tester",
        "full_name": "AI Tester",
        "email": f"ai_{uid}@example.com",
        "departement": "IT",
        "position": "IT Support Specialist",
    }
    r = client.post("/employees/", headers=admin_auth, json=payload)
    assert r.status_code == 201, r.text

    r = client.post(
        "/match/job",
        headers=admin_auth,
        json={
            "job_title": "IT Support Engineer",
            "required_skills": ["python", "sql"],
            "min_experience": 0,
            "limit": 10,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["results"], list)

def test_upload_cv_smoke(client, admin_auth, monkeypatch):
    monkeypatch.setattr(candidates_api, "extract_text", lambda *_: "Python SQL")
    monkeypatch.setattr(candidates_api, "detect_skills", lambda _: ["python", "sql"])

    files = {"file": ("resume.pdf", b"%PDF-1.4 fake", "application/pdf")}
    r = client.post("/candidates/upload_cv", headers=admin_auth, files=files)
    assert r.status_code == 200, r.text


