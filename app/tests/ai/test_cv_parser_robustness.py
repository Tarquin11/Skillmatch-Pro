import pytest
from app.schemas.candidate import CandidateUploadRespose
from app.services import cv_parser


KNOWN_SKILLS = ["python", "sql", "docker", "java", "react"]


def _assert_safe_contract(payload: dict) -> None:
    expected_keys = {
        "filename",
        "ok",
        "degraded",
        "errors",
        "warnings",
        "text_length",
        "skills",
        "preview",
        "extracted_skills",
        "predicted_title",
        "predicted_experience_years",
    }
    assert set(payload.keys()) == expected_keys
    assert isinstance(payload["ok"], bool)
    assert isinstance(payload["degraded"], bool)
    assert isinstance(payload["errors"], list)
    assert isinstance(payload["warnings"], list)
    assert isinstance(payload["skills"], list)
    assert isinstance(payload["preview"], str)
    assert isinstance(payload["text_length"], int)
    assert payload["text_length"] >= 0
    assert isinstance(payload["extracted_skills"], list)
    # Validate schema-level compatibility for API responses.
    CandidateUploadRespose(**payload)


def _parse_never_crash(**kwargs) -> dict:
    try:
        return cv_parser.parse_cv_safe(**kwargs)
    except Exception as exc:  # pragma: no cover - failure path guard
        pytest.fail(f"parse_cv_safe crashed unexpectedly: {exc!r}")


@pytest.mark.parametrize(
    "text",
    [
        "Pythn devloper with 3 yers expreince in SQl and Dockr",
        "John Doe\njohn@example.com\n+21600000000\nTunisia",
        "S K I L L S\nP Y T H O N , S Q L\nE X P E R I E N C E\n2 0 2 2 - 2 0 2 4",
        "Experience\nSoftware Engin",
    ],
)
def test_parse_cv_safe_handles_noisy_inputs_without_crash(monkeypatch, text):
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_: text)
    payload = _parse_never_crash(
        file_bytes=b"fake",
        filename="resume.pdf",
        known_skills=KNOWN_SKILLS,
        min_confidence=0.6,
        use_semantic=False,
    )
    _assert_safe_contract(payload)
    assert payload["ok"] is True
    assert payload["text_length"] == len(text)


def test_parse_cv_safe_empty_text_degrades_gracefully(monkeypatch):
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_: "   ")
    payload = _parse_never_crash(
        file_bytes=b"fake",
        filename="empty.pdf",
        known_skills=KNOWN_SKILLS,
    )
    _assert_safe_contract(payload)
    assert payload["ok"] is True
    assert payload["degraded"] is True
    assert "empty_text" in payload["warnings"]
    assert payload["skills"] == []


def test_parse_cv_safe_extraction_failure_never_raises(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise ValueError("bad file")

    monkeypatch.setattr(cv_parser, "extract_text", _boom)
    payload = _parse_never_crash(
        file_bytes=b"broken",
        filename="broken.pdf",
        known_skills=KNOWN_SKILLS,
    )
    _assert_safe_contract(payload)
    assert payload["ok"] is False
    assert payload["degraded"] is True
    assert any(err.startswith("extract_text:ValueError") for err in payload["errors"])


def test_parse_cv_safe_stage_failures_degrade_not_crash(monkeypatch):
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_: "Python developer with 2 years experience in SQL.")

    def _title_boom(*_args, **_kwargs):
        raise RuntimeError("title parser error")

    def _exp_boom(*_args, **_kwargs):
        raise RuntimeError("experience parser error")

    monkeypatch.setattr(cv_parser, "detect_title", _title_boom)
    monkeypatch.setattr(cv_parser, "detect_experience_years", _exp_boom)

    payload = _parse_never_crash(
        file_bytes=b"ok",
        filename="resume.pdf",
        known_skills=KNOWN_SKILLS,
    )
    _assert_safe_contract(payload)
    assert payload["ok"] is True
    assert payload["degraded"] is True
    assert any(err.startswith("detect_title:RuntimeError") for err in payload["errors"])
    assert any(err.startswith("detect_experience_years:RuntimeError") for err in payload["errors"])


def test_parse_cv_safe_legacy_fallback_when_primary_skill_detector_fails(monkeypatch):
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_: "Python SQL")

    def _skills_boom(*_args, **_kwargs):
        raise RuntimeError("primary skill detector down")

    monkeypatch.setattr(cv_parser, "detect_skills_with_confidence", _skills_boom)
    monkeypatch.setattr(cv_parser, "detect_skills", lambda *_args, **_kwargs: ["python", "sql"])

    payload = _parse_never_crash(
        file_bytes=b"ok",
        filename="resume.pdf",
        known_skills=KNOWN_SKILLS,
    )
    _assert_safe_contract(payload)
    assert payload["degraded"] is True
    assert payload["skills"] == ["python", "sql"]
    assert "legacy_skill_fallback" in payload["warnings"]
    assert all(row["source"] == "legacy" for row in payload["extracted_skills"])


def test_parse_cv_safe_when_both_skill_detectors_fail(monkeypatch):
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_: "Python SQL")

    def _skills_boom(*_args, **_kwargs):
        raise RuntimeError("primary skill detector down")

    def _legacy_boom(*_args, **_kwargs):
        raise RuntimeError("legacy skill detector down")

    monkeypatch.setattr(cv_parser, "detect_skills_with_confidence", _skills_boom)
    monkeypatch.setattr(cv_parser, "detect_skills", _legacy_boom)

    payload = _parse_never_crash(
        file_bytes=b"ok",
        filename="resume.pdf",
        known_skills=KNOWN_SKILLS,
    )
    _assert_safe_contract(payload)
    assert payload["degraded"] is True
    assert payload["skills"] == []
    assert any(err.startswith("detect_skills_with_confidence:RuntimeError") for err in payload["errors"])
    assert any(err.startswith("detect_skills_legacy:RuntimeError") for err in payload["errors"])


def test_parse_cv_safe_with_real_corrupted_pdf_bytes_never_crashes():
    payload = _parse_never_crash(
        file_bytes=b"%PDF-1.7\n%%%%broken%%%%\x00\xffgarbage",
        filename="corrupted.pdf",
        known_skills=KNOWN_SKILLS,
        min_confidence=0.6,
        use_semantic=False,
    )
    _assert_safe_contract(payload)
    # Extraction can fail; robustness requirement is no crash and valid response.
    assert payload["degraded"] is True or payload["ok"] is True


def test_parse_cv_safe_caps_max_text_length(monkeypatch):
    very_long = "A" * (cv_parser.MAX_TEXT_CHARS + 500)
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_: very_long)

    payload = _parse_never_crash(
        file_bytes=b"ok",
        filename="long.pdf",
        known_skills=KNOWN_SKILLS,
    )
    _assert_safe_contract(payload)
    assert payload["text_length"] == cv_parser.MAX_TEXT_CHARS
    assert payload["degraded"] is True
    assert "text_truncated" in payload["warnings"]


def test_parse_cv_safe_handles_invalid_file_bytes_type(monkeypatch):
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_: "Python SQL")
    payload = _parse_never_crash(
        file_bytes="not-bytes",  # type: ignore[arg-type]
        filename="resume.pdf",
        known_skills=KNOWN_SKILLS,
    )
    _assert_safe_contract(payload)
    assert payload["degraded"] is True
    assert "invalid_file_bytes_type" in payload["warnings"]


def test_parse_cv_safe_budget_hit_with_skills_is_warning_only(monkeypatch):
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_: "Python SQL developer with practical projects")
    monkeypatch.setattr(
        cv_parser,
        "detect_skills_with_confidence",
        lambda **_kwargs: [{"skill": "python", "confidence": 0.91, "source": "exact"}],
    )
    ticks = iter([10.0, 10.0 + cv_parser.DEFAULT_SKILL_TIME_BUDGET_SECONDS + 0.05])
    monkeypatch.setattr(cv_parser.time, "perf_counter", lambda: next(ticks))

    payload = _parse_never_crash(
        file_bytes=b"ok",
        filename="resume.pdf",
        known_skills=KNOWN_SKILLS,
    )
    _assert_safe_contract(payload)
    assert "skills_time_budget_hit" in payload["warnings"]
    assert payload["skills"] == ["python"]
    assert payload["degraded"] is False


def test_parse_cv_safe_budget_hit_without_skills_marks_degraded(monkeypatch):
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_: "General profile text with little technical detail")
    monkeypatch.setattr(cv_parser, "detect_skills_with_confidence", lambda **_kwargs: [])
    ticks = iter([20.0, 20.0 + cv_parser.DEFAULT_SKILL_TIME_BUDGET_SECONDS + 0.05])
    monkeypatch.setattr(cv_parser.time, "perf_counter", lambda: next(ticks))

    payload = _parse_never_crash(
        file_bytes=b"ok",
        filename="resume.pdf",
        known_skills=KNOWN_SKILLS,
    )
    _assert_safe_contract(payload)
    assert "skills_time_budget_hit" in payload["warnings"]
    assert "no_skills_detected" in payload["warnings"]
    assert payload["skills"] == []
    assert payload["degraded"] is True
