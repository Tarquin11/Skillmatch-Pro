"""Negation, hedged phrasing, and cross-skill normalization."""

from app.services import cv_parser
from app.services.cv_parser import apply_weak_hedge_penalty_to_rows, attach_confidence_normalized, parse_cv_safe


def test_span_negation_english_and_french():
    assert cv_parser._span_text_negated("No experience in project management whatsoever")
    assert cv_parser._span_text_negated("Sans experience en gestion de projet")
    assert cv_parser._span_text_negated("Jamais travaille sur du budgeting")
    assert not cv_parser._span_text_negated("Led project management office for three years")


def test_skill_negation_scope_across_multiline_window():
    text = (
        "Experience\n"
        "No hands-on experience with\n"
        "project management\n"
        "Built Python automation for reporting\n"
    )
    assert cv_parser._skill_negated_in_text("project management", text, window_lines=1)
    assert not cv_parser._skill_negated_in_text("python", text, window_lines=1)


def test_skill_negation_scope_french_ne_pas_pattern():
    text = (
        "Je ne maitrise pas la gestion de projet complexe\n"
        "Python et SQL utilises en stage\n"
    )
    assert cv_parser._skill_negated_in_text("project management", text, window_lines=1)
    assert not cv_parser._skill_negated_in_text("python", text, window_lines=1)


def test_weak_hedge_in_evidence():
    assert cv_parser._evidence_weak_hedge("basic understanding of python and sql")
    assert cv_parser._evidence_weak_hedge("familiar with budgeting concepts")
    assert cv_parser._evidence_weak_hedge("exposed to agile ceremonies")
    assert not cv_parser._evidence_weak_hedge("Owned the full budgeting process for the program")


def test_weak_hedge_penalty_and_normalization():
    rows = [
        {"skill": "python", "confidence": 0.9, "source": "exact", "evidence": ["basic understanding of python"]},
        {"skill": "sql", "confidence": 0.8, "source": "exact", "evidence": ["daily sql reporting"]},
    ]
    apply_weak_hedge_penalty_to_rows(rows)
    assert rows[0]["confidence"] < 0.9
    assert rows[1]["confidence"] == 0.8
    attach_confidence_normalized(rows)
    assert max(r["confidence_normalized"] for r in rows) == 1.0
    assert rows[0]["confidence_normalized"] == 1.0
    assert rows[1]["confidence_normalized"] < 1.0


def test_parse_cv_safe_sets_confidence_normalized(monkeypatch):
    text = "SKILLS\nPython\nSQL\n"
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_a, **_k: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=["python", "sql"],
        min_confidence=0.55,
        use_semantic=False,
        use_semantic_augment=False,
    )
    norms = [float(r["confidence_normalized"]) for r in payload["extracted_skills"]]
    assert norms
    assert max(norms) == 1.0


def test_parse_cv_safe_filters_negated_skills_beyond_semantic(monkeypatch):
    text = (
        "SKILLS\n"
        "No experience with project management\n"
        "Python\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_a, **_k: text)
    monkeypatch.setattr(cv_parser, "extract_open_vocabulary_skill_rows", lambda **_kwargs: [])
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=["python", "project management"],
        min_confidence=0.55,
        use_semantic=False,
        use_semantic_augment=False,
    )
    skills = {str(s).strip().lower() for s in payload["skills"]}
    assert "python" in skills
    assert "project management" not in skills
    assert "negated_skills_filtered" in payload["warnings"]


def test_merge_skill_rows_fuses_mentions_and_evidences():
    catalog_rows = [
        {
            "skill": "python",
            "confidence": 0.82,
            "source": "exact:skills",
            "evidence": ["Python"],
        }
    ]
    extra_rows = [
        {
            "skill": "python",
            "confidence": 0.79,
            "source": "sentence_text",
            "evidence": ["used python in internship"],
        },
        {
            "skill": "python",
            "confidence": 0.74,
            "source": "semantic_augment",
            "evidence": ["built data pipeline in python"],
        },
    ]

    merged = cv_parser._merge_skill_rows(catalog_rows, extra_rows)
    assert len(merged) == 1
    row = merged[0]
    assert row["skill"] == "python"
    assert float(row["confidence"]) > 0.82
    assert {"catalog", "sentence_text", "semantic_augment"}.issubset(set(row.get("_conf_channels", set())))
    evidences = row.get("evidence") or []
    assert any("internship" in e for e in evidences)
    assert any("pipeline" in e for e in evidences)
