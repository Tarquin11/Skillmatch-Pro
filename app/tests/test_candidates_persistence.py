import re
import uuid

from app.api import candidates as candidates_api


def _fake_parsed_payload(**overrides) -> dict:
    payload = {
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
            "certification": ["AWS Certified Developer"],
            "hands_on_project": ["Built an internal applicant tracking dashboard using React and FastAPI"],
            "project_text": [],
        },
        "extracted_skills": [
            {"skill": "python", "confidence": 0.99, "confidence_normalized": 1.0, "source": "exact", "evidence": []},
            {"skill": "sql", "confidence": 0.98, "confidence_normalized": 0.99, "source": "exact", "evidence": []},
            {"skill": "docker", "confidence": 0.97, "confidence_normalized": 0.98, "source": "exact", "evidence": []},
        ],
        "preview": "Backend engineer profile",
        "extracted_full_name": "Sara Ben Ali",
        "extracted_email": "sara.ben.ali@example.com",
        "extracted_phone": "+216 54 111 222",
        "predicted_title": "Backend Engineer",
        "predicted_experience_years": 3.0,
        "certifications": ["AWS Certified Developer"],
        "hands_on_projects": ["Built an internal applicant tracking dashboard using React and FastAPI"],
    }
    payload.update(overrides)
    return payload


def test_upload_cv_autosaves_profile_and_lists_in_candidates(client, admin_auth, monkeypatch):
    before = client.get("/candidates/", headers=admin_auth, params={"limit": 100}).json()
    before_count = len(before)

    marker = uuid.uuid4().hex[:8]
    monkeypatch.setattr(
        candidates_api,
        "parse_cv_safe",
        lambda **_: _fake_parsed_payload(
            extracted_full_name="Sara Ben Ali",
            extracted_email=f"sara.ben.ali.{marker}@example.com",
            extracted_phone="+216 54 111 222",
        ),
        raising=False,
    )
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
    assert first["full_name"] == "Sara Ben Ali"
    assert first["email"] == f"sara.ben.ali.{marker}@example.com"
    assert first["phone"] == "+216 54 111 222"
    assert first["predicted_title"] == "Backend Engineer"
    assert set(first["skills"]) >= {"python", "sql", "docker"}
    assert first["certifications"] == ["AWS Certified Developer"]
    assert first["hands_on_projects"] == ["Built an internal applicant tracking dashboard using React and FastAPI"]

    search = client.get("/candidates/", headers=admin_auth, params={"search": marker, "limit": 20})
    assert search.status_code == 200, search.text
    search_rows = search.json()
    assert len(search_rows) >= 1


def test_update_candidate_name_id_and_skills(client, admin_auth, monkeypatch):
    marker = uuid.uuid4().hex[:8]
    monkeypatch.setattr(
        candidates_api,
        "parse_cv_safe",
        lambda **_: _fake_parsed_payload(extracted_email=f"joanne.cain.{marker}@example.com"),
        raising=False,
    )
    filename = f"joanne_cain_{marker}_resume.pdf"
    files = {"file": (filename, b"%PDF-1.7\n%fake cv content", "application/pdf")}
    upload = client.post("/candidates/upload_cv", headers=admin_auth, files=files)
    assert upload.status_code == 200, upload.text

    listed = client.get("/candidates/", headers=admin_auth, params={"search": marker, "limit": 20})
    assert listed.status_code == 200, listed.text
    rows = listed.json()
    assert len(rows) >= 1
    candidate = rows[0]

    patched = client.patch(
        f"/candidates/{candidate['id']}",
        headers=admin_auth,
        json={
            "full_name": "Joanne Cain",
            "employee_number": f"CAND-{marker.upper()}",
            "skills": ["html", "css", "javascript", "html"],
            "certifications": ["Azure Fundamentals", "AWS Practitioner", "Azure Fundamentals"],
            "hands_on_projects": [
                "Built an internship matching portal with Angular and FastAPI",
                "Automated CV parsing benchmark runner",
            ],
        },
    )
    assert patched.status_code == 200, patched.text
    updated = patched.json()
    assert updated["full_name"] == "Joanne Cain"
    assert updated["employee_number"] == f"CAND-{marker.upper()}"
    assert set(updated["skills"]) == {"html", "css", "javascript"}
    assert updated["certifications"] == ["Azure Fundamentals", "AWS Practitioner"]
    assert updated["hands_on_projects"] == [
        "Built an internship matching portal with Angular and FastAPI",
        "Automated CV parsing benchmark runner",
    ]


def test_update_candidate_rejects_duplicate_employee_number(client, admin_auth, monkeypatch):
    files_a = {"file": ("alpha_resume.pdf", b"%PDF-1.7\n%fake cv content", "application/pdf")}
    files_b = {"file": ("beta_resume.pdf", b"%PDF-1.7\n%fake cv content", "application/pdf")}
    monkeypatch.setattr(
        candidates_api,
        "parse_cv_safe",
        lambda **_: _fake_parsed_payload(extracted_email=f"{uuid.uuid4().hex[:8]}@example.com"),
        raising=False,
    )
    upload_a = client.post("/candidates/upload_cv", headers=admin_auth, files=files_a)
    upload_b = client.post("/candidates/upload_cv", headers=admin_auth, files=files_b)
    assert upload_a.status_code == 200, upload_a.text
    assert upload_b.status_code == 200, upload_b.text

    listed = client.get("/candidates/", headers=admin_auth, params={"sort_by": "id", "sort_dir": "desc", "limit": 20})
    assert listed.status_code == 200, listed.text
    rows = listed.json()
    assert len(rows) >= 2
    newer = rows[0]
    older = rows[1]

    conflict = client.patch(
        f"/candidates/{newer['id']}",
        headers=admin_auth,
        json={"employee_number": older["employee_number"]},
    )
    assert conflict.status_code == 409, conflict.text
    body = conflict.json()
    assert body["detail"]["code"] == "employee_number_already_exists"


def test_upload_cv_falls_back_to_preview_contact_details(client, admin_auth, monkeypatch):
    marker = uuid.uuid4().hex[:8]
    preview = (
        "Anissa Ben Salem\n"
        f"Tunis, Tunisia • +216 22 000 111 • anissa.ben.salem.{marker}@example.com\n"
        "Education\n"
    )
    monkeypatch.setattr(
        candidates_api,
        "parse_cv_safe",
        lambda **_: _fake_parsed_payload(
            extracted_full_name=None,
            extracted_email=None,
            extracted_phone=None,
            preview=preview,
        ),
        raising=False,
    )
    files = {"file": (f"anissa_{marker}.pdf", b"%PDF-1.7\n%fake cv content", "application/pdf")}
    upload = client.post("/candidates/upload_cv", headers=admin_auth, files=files)
    assert upload.status_code == 200, upload.text

    listed = client.get("/candidates/", headers=admin_auth, params={"search": marker, "limit": 20})
    assert listed.status_code == 200, listed.text
    rows = listed.json()
    assert rows, "Expected candidate row from preview fallback"
    first = rows[0]
    assert first["full_name"] == "Anissa Ben Salem"
    assert first["email"] == f"anissa.ben.salem.{marker}@example.com"
    assert first["phone"] == "+216 22 000 111"


def test_upload_cv_falls_back_to_obfuscated_preview_contact_details(client, admin_auth, monkeypatch):
    marker = uuid.uuid4().hex[:8]
    preview = (
        "Nour El Houda Souissi\n"
        f"Email: nour.el.houda.{marker} [at] outlook [dot] com\n"
        "Mobile: +216 5 4 1 4 2 3 1 6\n"
    )
    monkeypatch.setattr(
        candidates_api,
        "parse_cv_safe",
        lambda **_: _fake_parsed_payload(
            extracted_full_name=None,
            extracted_email=None,
            extracted_phone=None,
            preview=preview,
        ),
        raising=False,
    )
    files = {"file": (f"nour_{marker}.pdf", b"%PDF-1.7\n%fake cv content", "application/pdf")}
    upload = client.post("/candidates/upload_cv", headers=admin_auth, files=files)
    assert upload.status_code == 200, upload.text

    listed = client.get("/candidates/", headers=admin_auth, params={"search": marker, "limit": 20})
    assert listed.status_code == 200, listed.text
    rows = listed.json()
    assert rows, "Expected candidate row from obfuscated preview fallback"
    first = rows[0]
    assert first["email"] == f"nour.el.houda.{marker}@outlook.com"
    assert re.sub(r"\D", "", str(first.get("phone") or "")) == "21654142316"


def test_upload_cv_aliases_duplicate_extracted_email(client, admin_auth, monkeypatch):
    marker = uuid.uuid4().hex[:8]
    duplicate_email = f"duplicate.{marker}@example.com"
    monkeypatch.setattr(
        candidates_api,
        "parse_cv_safe",
        lambda **_: _fake_parsed_payload(extracted_email=duplicate_email),
        raising=False,
    )

    first_upload = client.post(
        "/candidates/upload_cv",
        headers=admin_auth,
        files={"file": (f"dup_a_{marker}.pdf", b"%PDF-1.7\n%fake cv content", "application/pdf")},
    )
    second_upload = client.post(
        "/candidates/upload_cv",
        headers=admin_auth,
        files={"file": (f"dup_b_{marker}.pdf", b"%PDF-1.7\n%fake cv content", "application/pdf")},
    )
    assert first_upload.status_code == 200, first_upload.text
    assert second_upload.status_code == 200, second_upload.text

    listed = client.get("/candidates/", headers=admin_auth, params={"search": marker, "limit": 20})
    assert listed.status_code == 200, listed.text
    emails = [str(row.get("email", "")).lower() for row in listed.json()]
    assert any(email == duplicate_email for email in emails)
    assert any(email.startswith(f"duplicate.{marker}+cv") for email in emails)


def test_delete_candidate_removes_profile(client, admin_auth, monkeypatch):
    marker = uuid.uuid4().hex[:8]
    monkeypatch.setattr(
        candidates_api,
        "parse_cv_safe",
        lambda **_: _fake_parsed_payload(extracted_email=f"delete.me.{marker}@example.com"),
        raising=False,
    )
    filename = f"delete_me_{marker}_resume.pdf"
    files = {"file": (filename, b"%PDF-1.7\n%fake cv content", "application/pdf")}
    upload = client.post("/candidates/upload_cv", headers=admin_auth, files=files)
    assert upload.status_code == 200, upload.text

    listed = client.get("/candidates/", headers=admin_auth, params={"search": marker, "limit": 20})
    assert listed.status_code == 200, listed.text
    rows = listed.json()
    assert rows, "Expected uploaded candidate in list"
    candidate_id = int(rows[0]["id"])

    deleted = client.delete(f"/candidates/{candidate_id}", headers=admin_auth)
    assert deleted.status_code == 204, deleted.text

    after = client.get("/candidates/", headers=admin_auth, params={"search": marker, "limit": 20})
    assert after.status_code == 200, after.text
    assert not any(int(row["id"]) == candidate_id for row in after.json())

    missing = client.delete(f"/candidates/{candidate_id}", headers=admin_auth)
    assert missing.status_code == 404, missing.text
    assert missing.json()["detail"]["code"] == "candidate_not_found"
