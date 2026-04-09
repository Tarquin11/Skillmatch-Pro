import uuid

from app.api import candidates as candidates_api
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
        "predicted_title",
        "predicted_experience_years",
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


def test_upload_cv_enriches_skill_ids_from_catalog(client, admin_auth, monkeypatch):
    suffix = uuid.uuid4().hex[:6]
    skill_python = f"Python Taxonomy {suffix}"
    skill_sql = f"SQL Taxonomy {suffix}"
    sid_python = client.post("/skills/", headers=admin_auth, json={"name": skill_python}).json()["id"]
    sid_sql = client.post("/skills/", headers=admin_auth, json={"name": skill_sql}).json()["id"]

    fake_parsed = {
        "ok": True,
        "degraded": False,
        "errors": [],
        "warnings": [],
        "text_length": 120,
        "skills": [skill_python, skill_sql, "graphql"],
        "skills_grouped": {"technical": [skill_python, skill_sql, "graphql"], "management": [], "business": [], "soft-skills": [], "other": []},
        "skill_hierarchy": [],
        "skill_graph": {},
        "extracted_languages": [],
        "language_details": [],
        "extraction_channels": {
            "catalog_match": [skill_python, skill_sql],
            "open_vocab": ["graphql"],
            "soft_skill": [],
            "sentence": [],
            "semantic_augment": [],
            "language": [],
            "project_text": [],
        },
        "preview": "profile",
        "extracted_skills": [
            {"skill": skill_python, "confidence": 0.92, "confidence_normalized": 1.0, "source": "lexicon", "evidence": []},
            {"skill": skill_sql, "confidence": 0.88, "confidence_normalized": 0.96, "source": "lexicon", "evidence": []},
            {"skill": "graphql", "confidence": 0.81, "confidence_normalized": 0.88, "source": "open_vocab", "evidence": []},
        ],
        "predicted_title": "Backend Engineer",
        "predicted_experience_years": 2.0,
    }
    monkeypatch.setattr(candidates_api, "parse_cv_safe", lambda **_: fake_parsed, raising=False)

    files = {"file": ("resume.pdf", b"%PDF-1.4 fake", "application/pdf")}
    r = client.post("/candidates/upload_cv", headers=admin_auth, files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    _assert_upload_contract(body)

    by_skill = {row["skill"]: row for row in body["extracted_skills"]}
    assert by_skill[skill_python]["skill_id"] == sid_python
    assert by_skill[skill_sql]["skill_id"] == sid_sql
    assert by_skill["graphql"]["skill_id"] is None


def test_upload_cv_uses_feature_flags_and_logs_metrics(client, admin_auth, monkeypatch):
    captured_kwargs: dict = {}

    def _fake_parse(**kwargs):
        captured_kwargs.update(kwargs)
        return {
            "ok": True,
            "degraded": False,
            "errors": [],
            "warnings": [],
            "text_length": 80,
            "skills": [],
            "skills_grouped": {"technical": [], "management": [], "business": [], "soft-skills": [], "other": []},
            "skill_hierarchy": [],
            "skill_graph": {},
            "extracted_languages": [],
            "language_details": [],
            "extraction_channels": {
                "catalog_match": [],
                "open_vocab": [],
                "soft_skill": [],
                "sentence": [],
                "semantic_augment": [],
                "language": [],
                "project_text": [],
            },
            "preview": "ok",
            "extracted_skills": [],
            "predicted_title": None,
            "predicted_experience_years": None,
        }

    metrics_events: list[dict] = []
    monkeypatch.setattr(candidates_api, "parse_cv_safe", _fake_parse, raising=False)
    monkeypatch.setattr(candidates_api, "log_structured_event", lambda *_a, **kw: metrics_events.append(kw), raising=False)

    monkeypatch.setattr(candidates_api.settings, "CV_PARSER_MIN_CONFIDENCE", 0.55, raising=False)
    monkeypatch.setattr(candidates_api.settings, "CV_PARSER_USE_SEMANTIC", True, raising=False)
    monkeypatch.setattr(candidates_api.settings, "CV_PARSER_USE_HF_NER", False, raising=False)
    monkeypatch.setattr(candidates_api.settings, "CV_PARSER_USE_SEMANTIC_AUGMENT", False, raising=False)
    monkeypatch.setattr(candidates_api.settings, "CV_PARSER_SKILL_TIME_BUDGET_SECONDS", 0.33, raising=False)
    monkeypatch.setattr(candidates_api.settings, "CV_PARSER_MODEL_VERSION", "cv_parser_ops_v1", raising=False)
    monkeypatch.setattr(candidates_api.settings, "CV_PARSER_SLO_MS", 5, raising=False)

    state = {"n": 0}

    def _fake_perf_counter() -> float:
        state["n"] += 1
        # Every call advances by 30ms so any measured segment is >= 30ms.
        return 100.0 + (0.03 * state["n"])

    monkeypatch.setattr(candidates_api.time, "perf_counter", _fake_perf_counter)

    files = {"file": ("resume.pdf", b"%PDF-1.4 fake", "application/pdf")}
    r = client.post("/candidates/upload_cv", headers=admin_auth, files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    _assert_upload_contract(body)
    assert "parse_slo_exceeded" in body["warnings"]

    assert captured_kwargs["min_confidence"] == 0.55
    assert captured_kwargs["use_semantic"] is True
    assert captured_kwargs["use_hf_ner"] is False
    assert captured_kwargs["use_semantic_augment"] is False
    assert captured_kwargs["skill_time_budget_seconds"] == 0.33

    metric_logs = [e for e in metrics_events if e.get("event") == "cv_parse_metrics"]
    assert metric_logs, "expected cv_parse_metrics structured log"
    assert metric_logs[-1]["parser_model_version"] == "cv_parser_ops_v1"
    assert metric_logs[-1]["slo_violation"] is True
