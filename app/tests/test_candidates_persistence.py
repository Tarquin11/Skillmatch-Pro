import uuid

from app.api import candidates as candidates_api


def _fake_parsed_payload() -> dict:
    return {
        "ok": True,
        "degraded": False,
        "errors": [],
        "warnings": [],
        "text_length": 240,
        "skills": ["python", "sql", "docker"],
        "skills_grouped": {
            "technical": ["python", "sql", "docker"],
            "management": [],
            "business": [],
            "soft-skills": [],
            "other": [],
        },
        "skill_hierarchy": [],
        "skill_graph": {},
        "extracted_languages": ["english"],
        "language_details": [{"language": "english", "level": "B2", "source": "language_section:languages"}],
        "extraction_channels": {
            "catalog_match": ["python", "sql", "docker"],
            "open_vocab": [],
            "soft_skill": [],
            "sentence": [],
            "semantic_augment": [],
            "language": ["english"],
            "project_text": [],
        },
        "extracted_skills": [
            {"skill": "python", "confidence": 0.99, "confidence_normalized": 1.0, "source": "exact", "evidence": []},
            {"skill": "sql", "confidence": 0.98, "confidence_normalized": 0.99, "source": "exact", "evidence": []},
            {"skill": "docker", "confidence": 0.97, "confidence_normalized": 0.98, "source": "exact", "evidence": []},
        ],
        "preview": "Backend engineer profile",
        "predicted_title": "Backend Engineer",
        "predicted_experience_years": 3.0,
    }


def test_upload_cv_autosaves_profile_and_lists_in_candidates(client, admin_auth, monkeypatch):
    monkeypatch.setattr(candidates_api, "parse_cv_safe", lambda **_: _fake_parsed_payload(), raising=False)

    before = client.get("/candidates/", headers=admin_auth, params={"limit": 100}).json()
    before_count = len(before)

    marker = uuid.uuid4().hex[:8]
    filename = f"sara_ben_ali_{marker}_resume.pdf"
    files = {"file": (filename, b"%PDF-1.7\n%fake cv content", "application/pdf")}

    upload = client.post("/candidates/upload_cv", headers=admin_auth, files=files)
    assert upload.status_code == 200, upload.text
    body = upload.json()
    assert body["ok"] is True
    assert body["degraded"] is False

    after = client.get("/candidates/", headers=admin_auth, params={"limit": 100, "sort_by": "id", "sort_dir": "desc"})
    assert after.status_code == 200, after.text
    rows = after.json()
    assert len(rows) >= before_count + 1

    first = rows[0]
    assert first["predicted_title"] == "Backend Engineer"
    assert set(first["skills"]) >= {"python", "sql", "docker"}

    search = client.get("/candidates/", headers=admin_auth, params={"search": marker, "limit": 20})
    assert search.status_code == 200, search.text
    search_rows = search.json()
    assert len(search_rows) >= 1
