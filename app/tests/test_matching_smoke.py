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
    monkeypatch.setattr(
        candidates_api,
        "parse_cv_safe",
        lambda **_: {
            "filename": "resume.pdf",
            "ok": True,
            "degraded": False,
            "errors": [],
            "warnings": [],
            "text_length": 10,
            "skills": ["python", "sql"],
            "skills_grouped": {"technical": ["python", "sql"], "management": [], "business": [], "soft-skills": [], "other": []},
            "skill_hierarchy": [],
            "skill_graph": {},
            "extracted_languages": ["english"],
            "language_details": [{"language": "english", "level": "B2", "source": "language_section:languages"}],
            "extraction_channels": {
                "catalog_match": ["python", "sql"],
                "open_vocab": [],
                "soft_skill": [],
                "sentence": [],
                "semantic_augment": [],
                "language": ["english"],
                "project_text": [],
            },
            "preview": "Python SQL",
            "extracted_skills": [
                {"skill": "python", "confidence": 0.98, "confidence_normalized": 1.0, "source": "exact", "evidence": []},
                {"skill": "sql", "confidence": 0.98, "confidence_normalized": 1.0, "source": "exact", "evidence": []},
            ],
            "predicted_title": "Developer",
            "predicted_experience_years": 1.0,
        },
    )

    files = {"file": ("resume.pdf", b"%PDF-1.4 fake", "application/pdf")}
    r = client.post("/candidates/upload_cv", headers=admin_auth, files=files)
    assert r.status_code == 200, r.text


def test_match_job_can_scope_to_saved_candidates_only(client, admin_auth, monkeypatch):
    uid = uuid.uuid4().hex[:6]
    employee_payload = {
        "employeeNumber": f"EMP-{uid}",
        "first_name": "Scope",
        "last_name": "Employee",
        "full_name": "Scope Employee",
        "email": f"scope_{uid}@example.com",
        "departement": "IT",
        "position": "Backend Engineer",
    }
    emp = client.post("/employees/", headers=admin_auth, json=employee_payload)
    assert emp.status_code == 201, emp.text
    employee_id = int(emp.json()["id"])

    monkeypatch.setattr(
        candidates_api,
        "parse_cv_safe",
        lambda **_: {
            "filename": "candidate.pdf",
            "ok": True,
            "degraded": False,
            "errors": [],
            "warnings": [],
            "text_length": 10,
            "skills": ["python", "sql"],
            "skills_grouped": {"technical": ["python", "sql"], "management": [], "business": [], "soft-skills": [], "other": []},
            "skill_hierarchy": [],
            "skill_graph": {},
            "extracted_languages": ["english"],
            "language_details": [{"language": "english", "level": "B2", "source": "language_section:languages"}],
            "extraction_channels": {
                "catalog_match": ["python", "sql"],
                "open_vocab": [],
                "soft_skill": [],
                "sentence": [],
                "semantic_augment": [],
                "language": ["english"],
                "project_text": [],
            },
            "preview": "Python SQL",
            "extracted_skills": [
                {"skill": "python", "confidence": 0.98, "confidence_normalized": 1.0, "source": "exact", "evidence": []},
                {"skill": "sql", "confidence": 0.98, "confidence_normalized": 1.0, "source": "exact", "evidence": []},
            ],
            "predicted_title": "Backend Engineer",
            "predicted_experience_years": 1.0,
        },
    )
    files = {"file": (f"candidate_scope_{uid}.pdf", b"%PDF-1.4 fake", "application/pdf")}
    upload = client.post("/candidates/upload_cv", headers=admin_auth, files=files)
    assert upload.status_code == 200, upload.text

    r = client.post(
        "/match/job?candidate_scope=candidates",
        headers=admin_auth,
        json={
            "job_title": "Backend Engineer",
            "required_skills": ["python", "sql"],
            "min_experience": 0,
            "limit": 2000,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    ids = {int(row["employee_id"]) for row in body["results"]}
    assert employee_id not in ids


