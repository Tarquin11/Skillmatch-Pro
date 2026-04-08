"""Stress-style cases: messy layout, no headings, mixed FR/EN."""

from app.services import cv_parser
from app.services.cv_parser import parse_cv_safe


def test_parse_cv_safe_no_section_headings_messy_spacing(monkeypatch):
    text = (
        "lucas   leblanc\n"
        "charge  de  projets   ti\n"
        "weirdspacinghere   python    sql   docker\n"
        "- gerer le budget et les risques sur plusieurs projets\n"
        "- coordination equipe et planification\n"
        "english   french   mixed   ligne   sans   structure\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_a, **_k: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=["python", "sql", "docker"],
        min_confidence=0.55,
        use_semantic=False,
        use_hf_ner=False,
    )
    assert payload["ok"] is True
    names = {str(r["skill"]) for r in payload["extracted_skills"]}
    assert "python" in names or "sql" in names
    assert payload["text_length"] > 0


def test_parse_cv_safe_mixed_fr_en_bullets(monkeypatch):
    text = (
        "PROFILE / PROFIL\n"
        "• Project management / Gestion de projet\n"
        "• Risk management — gestion des risques\n"
        "• EN: fluent | FR: natif\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_a, **_k: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=[],
        min_confidence=0.6,
        use_semantic=False,
        use_hf_ner=False,
    )
    assert payload["ok"] is True
    skills = {str(r["skill"]) for r in payload["extracted_skills"]}
    assert "project management" in skills or "risk management" in skills


def test_parse_cv_safe_weird_pdf_tokens_and_letter_spaced_heading(monkeypatch):
    text = (
        "S K I L L S\n"
        "JavaScript,   React\n"
        "E X P E R I E N C E\n"
        "- Superviser   les   livrables   et   le   budget\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_a, **_k: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=["javascript", "react"],
        min_confidence=0.55,
        use_semantic=False,
        use_hf_ner=False,
    )
    assert payload["ok"] is True
    assert payload["text_length"] > 0
