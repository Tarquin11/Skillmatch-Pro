import json
from pathlib import Path

from app.services import cv_parser
from app.services.cv_parser import parse_cv_safe


def _fixture_path() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "cv_hard_cases.jsonl"


def _load_cases() -> list[dict]:
    rows: list[dict] = []
    for raw in _fixture_path().read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def test_cv_parser_hard_case_regressions(monkeypatch):
    known_skills = [
        "python",
        "sql",
        "docker",
        "react",
        "flask",
        "kali linux",
        "web exploitation",
    ]
    for case in _load_cases():
        monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, _t=str(case.get("text", "")): _t)
        payload = parse_cv_safe(
            file_bytes=b"%PDF-1.4\n",
            filename="case.pdf",
            known_skills=known_skills,
            min_confidence=0.6,
            use_semantic=False,
        )
        # Override with snippet semantics by using channels and language extraction output.
        skills = {str(s).strip().lower() for s in payload.get("skills", []) if s}
        langs = {str(s).strip().lower() for s in payload.get("extracted_languages", []) if s}

        for s in case.get("must_include_skills", []):
            assert s.lower() in skills, f"{case['id']}: missing expected skill {s!r}"
        for s in case.get("must_not_include_skills", []):
            assert s.lower() not in skills, f"{case['id']}: unexpected skill {s!r}"
        for l in case.get("must_include_languages", []):
            assert l.lower() in langs, f"{case['id']}: missing expected language {l!r}"
        for l in case.get("must_not_include_languages", []):
            assert l.lower() not in langs, f"{case['id']}: unexpected language {l!r}"
