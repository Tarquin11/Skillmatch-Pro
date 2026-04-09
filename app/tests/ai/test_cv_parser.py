from app.services import cv_parser
from app.services.cv_parser import (
    _extract_sections,
    detect_skill_spans_with_ensemble,
    detect_experience_years,
    detect_skills,
    detect_skills_with_confidence,
    detect_title,
    extract_text,
    parse_cv_safe,
)


def test_detect_skills_with_confidence_uses_known_skills():
    text = "Python, SQL, Project Management"
    known_skills = ["Python", "SQL", "Project Management", "React"]
    rows = detect_skills_with_confidence(text, known_skills=known_skills)
    found = {row["skill"] for row in rows}
    assert {"python", "sql", "project management"}.issubset(found)
    assert all(0.0 <= row["confidence"] <= 1.0 for row in rows)


def test_detect_skills_wrapper():
    out = detect_skills("Python", known_skills=["Python"])
    assert out == ["python"]


def test_detect_skills_avoids_short_acronym_false_positive():
    text = "I am a management student and mobile app builder."
    known_skills = ["Account Management", "Timeline Management", "A/B Testing"]
    rows = detect_skills_with_confidence(text, known_skills=known_skills, min_confidence=0.9, use_semantic=False)
    found = {row["skill"] for row in rows}
    assert "account management" not in found
    assert "timeline management" not in found
    assert "a/b testing" not in found


def test_detect_title_extracts_role_prefix():
    text = "Jane Doe\njane@example.com\nData Analyst with 3 years of experience in dashboards"
    assert detect_title(text) == "Data Analyst"


def test_detect_experience_years_from_explicit_pattern():
    text = "Backend Engineer with 4+ years experience in Python and APIs."
    assert detect_experience_years(text) == 4.0


def test_detect_experience_years_ignores_education_ranges():
    text = "Education\n2019-2022 University\nProjects\nBuilt recommendation API"
    assert detect_experience_years(text) is None


def test_detect_experience_years_from_keyword_context_years():
    text = (
        "February 2023\n"
        "Observation Internship\n"
        "Built process docs\n"
        "March 2025\n"
        "Working on web APIs\n"
    )
    assert detect_experience_years(text) == 2.0


def test_extract_text_unsupported_extension_returns_empty():
    assert extract_text(b"plain text bytes", "resume.unknown_ext") == ""


def test_extract_text_utf8_txt():
    out = extract_text("Ligne une\nLigne deux".encode("utf-8"), "snippet.txt")
    assert "ligne une" in out.lower()
    assert "ligne deux" in out.lower()


def test_detect_title_returns_none_for_empty_text():
    assert detect_title("") is None


def test_detect_skills_with_empty_known_skills_returns_empty():
    rows = detect_skills_with_confidence("Python SQL", known_skills=[])
    assert rows == []


def test_parse_cv_safe_open_vocab_from_skill_sections(monkeypatch):
    text = (
        "SKILLS\n"
        "- Python\n"
        "- Web Exploitation\n"
        "- React / Flask\n"
        "TOOLS\n"
        "- Burpsuite\n"
        "LANGUAGES\n"
        "- Arabic (Mother tongue)\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=["python"],
        min_confidence=0.6,
        use_semantic=False,
    )
    names = {str(r["skill"]) for r in payload["extracted_skills"]}
    assert "python" in names
    assert "web exploitation" in names
    assert "react" in names
    assert "flask" in names
    assert "burp suite" in names
    assert "arabic" in payload["extracted_languages"]
    assert payload["extraction_channels"]["language"] == ["arabic"]
    assert any("web exploitation" == s for s in payload["extraction_channels"]["open_vocab"])


def test_parse_cv_safe_letter_spaced_skill_heading_opens_vocab(monkeypatch):
    text = "S K I L L S\n- Kali Linux\n"
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=[],
        min_confidence=0.6,
        use_semantic=False,
    )
    names = {str(r["skill"]) for r in payload["extracted_skills"]}
    assert "kali linux" in names


def test_five_letter_spaced_heading_collapses_like_tools():
    text = "T O O L S\n- SQL\n- Docker\n"
    sections, weights = _extract_sections(text)
    assert "tools" in sections
    assert "sql" not in sections and "docker" not in sections
    assert any("sql" in ln.lower() for ln in sections.get("tools", []))


def test_open_vocab_rejects_date_and_project_sentences(monkeypatch):
    text = (
        "SKILLS\n"
        "- 5 January - 20 February 2025 web exploitation\n"
        "- implementation of a gps-based bus geolocation\n"
        "- Python\n"
        "LANGUAGES\n"
        "- French (B1 Level)\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=[],
        min_confidence=0.6,
        use_semantic=False,
    )
    names = {str(r["skill"]) for r in payload["extracted_skills"]}
    assert "python" in names
    assert "french" in payload["extracted_languages"]
    assert not any("january" in n for n in names)
    assert not any("implementation" in n for n in names)
    assert not any("geolocation" in n for n in names)
    assert not any("french" == n for n in names)


def test_parse_cv_safe_extracts_languages_with_french_heading(monkeypatch):
    text = (
        "COMPETENCES TECHNIQUES\n"
        "- Python\n"
        "LANGUES\n"
        "- Francais (B1)\n"
        "- Anglais (B2)\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=[],
        min_confidence=0.6,
        use_semantic=False,
    )
    assert payload["extracted_languages"] == ["french", "english"]
    assert payload["language_details"][0]["level"] == "B1"
    assert payload["language_details"][1]["level"] == "B2"


def test_language_fallback_pairs_cefr_to_nearest_following_language_on_line(monkeypatch):
    text = "Languages: Arabic (C1), Swedish (A1)"
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=[],
        min_confidence=0.6,
        use_semantic=False,
    )
    by_lang = {row["language"]: row.get("level") for row in payload["language_details"]}
    assert by_lang.get("arabic") == "C1"
    assert by_lang.get("swedish") == "A1"


def test_parse_cv_safe_extracts_languages_without_heading_fallback(monkeypatch):
    text = "Skills: Python, Docker\nFrench (B1 level), English (B2), Arabic"
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=[],
        min_confidence=0.6,
        use_semantic=False,
    )
    assert {"french", "english", "arabic"}.issubset(set(payload["extracted_languages"]))
    fr = [row for row in payload["language_details"] if row.get("language") == "french"]
    en = [row for row in payload["language_details"] if row.get("language") == "english"]
    assert fr and fr[0].get("level") == "B1"
    assert en and en[0].get("level") == "B2"


def test_language_fallback_avoids_education_false_positive(monkeypatch):
    text = "Baccalaureat Francais (equivalent DEC)\nUniversite de Paris"
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=[],
        min_confidence=0.6,
        use_semantic=False,
    )
    assert "french" not in payload["extracted_languages"]


def test_detect_skill_spans_with_hf_ner_path(monkeypatch):
    monkeypatch.setattr(
        cv_parser,
        "_extract_hf_ner_spans",
        lambda _text: ["JavaScript", "Burpsuite"],
    )
    rows = detect_skill_spans_with_ensemble(
        text="random text",
        known_skills=["javascript", "burp suite", "python"],
        min_confidence=0.6,
        use_hf_ner=True,
    )
    found = {str(r["skill"]) for r in rows}
    sources = {str(r["source"]) for r in rows}
    assert "javascript" in found
    assert any(src.startswith("ner_span:hf_ner") for src in sources)


def test_open_vocab_rejects_french_action_bullets(monkeypatch):
    text = (
        "COMPETENCES\n"
        "- Gerer un agenda et un budget\n"
        "- Transmettre l information a une equipe\n"
        "- Clients\n"
        "- Organisation\n"
        "- Leadership\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=[],
        min_confidence=0.6,
        use_semantic=False,
    )
    names = {str(r["skill"]) for r in payload["extracted_skills"]}
    assert "gerer un agenda et un budget" not in names
    assert "transmettre l information a une equipe" not in names
    assert "clients" not in names
    assert "organisation" not in names
    assert "leadership" in names


def test_open_vocab_keeps_managerial_soft_skills(monkeypatch):
    text = (
        "COMPETENCES\n"
        "- Leadership\n"
        "- Relation client\n"
        "- Capacite d adaptation\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=[],
        min_confidence=0.6,
        use_semantic=False,
    )
    names = {str(r["skill"]) for r in payload["extracted_skills"]}
    assert "leadership" in names
    assert "relation client" in names
    assert "capacite d adaptation" in names
    assert payload["extraction_channels"]["soft_skill"]
    soft_rows = [r for r in payload["extracted_skills"] if str(r.get("source", "")).startswith("softskill")]
    assert all(0.6 <= float(r["confidence"]) <= 0.9 for r in soft_rows)


def test_soft_skill_fallback_when_no_other_skills(monkeypatch):
    text = (
        "COMPETENCES\n"
        "- Gerer un agenda et un budget\n"
        "- Transmettre l information a une equipe\n"
        "- Coordination des equipes\n"
        "- Gestion de projet\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=[],
        min_confidence=0.6,
        use_semantic=False,
    )
    names = {str(r["skill"]) for r in payload["extracted_skills"]}
    assert "project management" in names or "gestion de projet" in names
    assert "coordination" in names or "coordination des equipes" in names
    assert len(payload["extraction_channels"]["soft_skill"]) >= 1


def test_detect_title_french_project_manager():
    text = "LUCAS LEBLANC\nCHARGE DE PROJETS TI\nMontreal, Canada"
    assert detect_title(text) == "CHARGE DE PROJETS TI"


def test_sentence_level_extraction_from_experience_bullets(monkeypatch):
    text = (
        "EXPERIENCE PROFESSIONNELLE\n"
        "- Gerer l echeancier, le budget et les risques\n"
        "- Coordonner les activites de l equipe\n"
        "- Planifier les projets\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=[],
        min_confidence=0.6,
        use_semantic=False,
    )
    names = {str(r["skill"]) for r in payload["extracted_skills"]}
    assert "planning" in names
    assert "budget management" in names
    assert "risk management" in names
    assert payload["extraction_channels"]["sentence"]


def test_clean_extracted_text_skips_noise_lines():
    cleaned = cv_parser._clean_extracted_text(
        "https://example.com\nmodeles-de-cv.com\nCOPYRIGHT 2024\nValid line\n"
    )
    assert "valid line" in cleaned.lower()
    assert "http" not in cleaned.lower()
    assert "copyright" not in cleaned.lower()


def test_parse_cv_safe_returns_grouping_and_hierarchy(monkeypatch):
    text = (
        "COMPETENCES\n"
        "- Gestion de projet\n"
        "- Planification\n"
        "- Gestion des risques\n"
        "- Gestion du budget\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=[],
        min_confidence=0.6,
        use_semantic=False,
    )
    assert "management" in payload["skills_grouped"]
    assert "project management" in payload["skills_grouped"]["management"]
    hierarchy_parents = {node["parent"] for node in payload["skill_hierarchy"]}
    assert "project management" in hierarchy_parents
    assert "project management" in payload["skill_graph"]
    assert payload["skill_graph"]["project management"]["children"]


def test_confidence_enrichment_spreads_scores(monkeypatch):
    text = (
        "EXPERIENCE\n"
        "- Gestion de projet et planification\n"
        "- Gestion de projet avec budget\n"
        "- Gestion des risques de projet\n"
        "- Leadership\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=[],
        min_confidence=0.6,
        use_semantic=False,
    )
    confidences = {
        str(r["skill"]): float(r["confidence"])
        for r in payload["extracted_skills"]
        if r.get("skill")
    }
    assert len(set(confidences.values())) > 1
    assert confidences.get("project management", 0) >= confidences.get("leadership", 0)
    for row in payload["extracted_skills"]:
        assert "evidence" in row
        assert "confidence_normalized" in row


def test_post_ocr_text_normalize_keeps_line_breaks():
    raw = "SKILLS\nPython    SQL\n-\nG e r e r   le  budget"
    out = cv_parser._post_ocr_text_normalize(raw)
    assert "\n" in out
    assert "    " not in out


def test_collect_skill_evidence_prefers_compact_snippet():
    text = "EXPERIENCE\n- Gérer le budget et les risques • Travail d'équipe avec clients\n"
    ev = cv_parser._collect_skill_evidence("budget management", text)
    assert ev
    assert all(len(x) <= 95 for x in ev)
    assert "budget" in ev[0].lower()


def test_semantic_augment_appends_when_enabled(monkeypatch):
    def fake_augment(**kwargs):
        return [
            {
                "skill": "graphql",
                "confidence": 0.68,
                "source": "semantic_augment",
                "evidence": ["designed typed API schemas for mobile clients"],
                "_conf_channels": {"semantic_augment"},
            }
        ]

    monkeypatch.setattr(cv_parser, "augment_skills_semantically_gated", fake_augment)
    text = "SKILLS\nPython\nEXPERIENCE\n- designed typed API schemas for mobile clients\n"
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_a, **_k: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=["python", "graphql"],
        min_confidence=0.55,
        use_semantic=False,
        use_semantic_augment=True,
    )
    names = {r["skill"] for r in payload["extracted_skills"]}
    assert "graphql" in names
    assert "graphql" in payload["extraction_channels"]["semantic_augment"]
    gql_rows = [r for r in payload["extracted_skills"] if r["skill"] == "graphql"]
    assert gql_rows[0]["source"] == "semantic_augment"
    assert gql_rows[0]["evidence"]
    assert "confidence_normalized" in gql_rows[0]


def test_semantic_augment_not_called_when_disabled(monkeypatch):
    called = {"n": 0}

    def fake_augment(**kwargs):
        called["n"] += 1
        return []

    monkeypatch.setattr(cv_parser, "augment_skills_semantically_gated", fake_augment)
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_a, **_k: "SKILLS\nPython\n")
    parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=["python"],
        use_semantic_augment=False,
    )
    assert called["n"] == 0
