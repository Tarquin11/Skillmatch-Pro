from app.services import cv_parser


def _assert_upload_contract(body: dict) -> None:
    expected = {
        "filename",
        "ok",
        "degraded",
        "errors",
        "warnings",
        "text_length",
        "skills",
        "skills_grouped",
        "skill_hierarchy",
        "skill_graph",
        "extracted_languages",
        "language_details",
        "extraction_channels",
        "preview",
        "extracted_skills",
        "extracted_full_name",
        "extracted_email",
        "extracted_phone",
        "predicted_title",
        "predicted_experience_years",
        "certifications",
        "hands_on_projects",
        "project_skill_links",
    }
    assert set(body.keys()) == expected
    assert isinstance(body["ok"], bool)
    assert isinstance(body["degraded"], bool)
    assert isinstance(body["errors"], list)
    assert isinstance(body["warnings"], list)
    assert isinstance(body["skills"], list)
    assert isinstance(body["skills_grouped"], dict)
    assert isinstance(body["skill_hierarchy"], list)
    assert isinstance(body["skill_graph"], dict)
    assert isinstance(body["extracted_languages"], list)
    assert isinstance(body["language_details"], list)
    assert isinstance(body["extraction_channels"], dict)
    assert isinstance(body["extracted_skills"], list)
    assert isinstance(body["certifications"], list)
    assert isinstance(body["hands_on_projects"], list)
    assert isinstance(body["project_skill_links"], list)
    assert isinstance(body["preview"], str)
    assert isinstance(body["text_length"], int)


def test_upload_cv_corrupted_pdf_no_crash(client, admin_auth):
    files = {"file": ("broken.pdf", b"%PDF-1.7\n%%%%broken%%%%\x00\xff", "application/pdf")}
    r = client.post("/candidates/upload_cv", headers=admin_auth, files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    _assert_upload_contract(body)
    assert body["degraded"] is True


def test_upload_cv_noisy_extracted_text_no_crash(client, admin_auth, monkeypatch):
    noisy = "S K I L L S\nP Y T H O N, S Q L\nW o r k E x p e r i e n c e\n2 0 2 2 - 2 0 2 4"
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_: noisy)

    files = {"file": ("resume.pdf", b"%PDF-1.4 fake", "application/pdf")}
    r = client.post("/candidates/upload_cv", headers=admin_auth, files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    _assert_upload_contract(body)
    # May find or miss skills depending on known skills in test DB; both are acceptable.
    assert body["ok"] in (True, False)


def test_upload_cv_partial_pdf_like_input_no_crash(client, admin_auth):
    files = {"file": ("partial.pdf", b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog", "application/pdf")}
    r = client.post("/candidates/upload_cv", headers=admin_auth, files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    _assert_upload_contract(body)
