import re

from app.services import cv_parser
from app.services.cv_parser import (
    _extract_sections,
    detect_experience_years,
    detect_skills,
    detect_skills_with_confidence,
    detect_title,
    extract_text,
    parse_cv_safe,
)


def test_detect_skills_with_confidence_boosts_exact_matches():
    text = "Python programming and machine learning"
    known_skills = ["Python", "Machine Learning"]
    rows = detect_skills_with_confidence(text, known_skills=known_skills)
    python_row = next(row for row in rows if row["skill"] == "python")
    ml_row = next(row for row in rows if row["skill"] == "ml")
    # Exact match should have higher confidence
    assert python_row["confidence"] >= 0.90
    assert ml_row["confidence"] >= 0.90


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


def test_parse_cv_safe_extracts_contact_details(monkeypatch):
    text = (
        "Mohamed Aziz Akrout\n"
        "Junior Penetration Tester specializing in Active Directory and Web Application Security\n"
        "Tunis, Tunisia • +216 54 142 316 • mohamedazizakrout@gmail.com • LinkedIn • GitHub\n"
        "Education\n"
        "Esprit School of Engineering\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=["python"],
        min_confidence=0.6,
        use_semantic=False,
    )
    assert payload["extracted_full_name"] == "Mohamed Aziz Akrout"
    assert payload["extracted_email"] == "mohamedazizakrout@gmail.com"
    assert payload["extracted_phone"] == "+216 54 142 316"


def test_parse_cv_safe_extracts_name_from_contact_line_with_separator(monkeypatch):
    text = (
        "MOHAMED AZIZ AKROUT | Junior Penetration Tester\n"
        "Tunis, Tunisia • +216 54 142 316 • mohamedazizakrout@gmail.com\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=[],
        min_confidence=0.6,
        use_semantic=False,
    )
    assert payload["extracted_full_name"] == "MOHAMED AZIZ AKROUT"
    assert payload["extracted_email"] == "mohamedazizakrout@gmail.com"
    assert payload["extracted_phone"] == "+216 54 142 316"


def test_parse_cv_safe_inferrs_name_from_email_when_header_is_noisy(monkeypatch):
    text = (
        "Curriculum Vitae\n"
        "Junior QA profile\n"
        "Contact: anissa.ben-salem@example.com\n"
        "Phone: +216 22 000 111\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=[],
        min_confidence=0.6,
        use_semantic=False,
    )
    assert payload["extracted_full_name"] == "Anissa Ben Salem"
    assert payload["extracted_email"] == "anissa.ben-salem@example.com"
    assert payload["extracted_phone"] == "+216 22 000 111"


def test_parse_cv_safe_extracts_obfuscated_email_and_spaced_phone(monkeypatch):
    text = (
        "Nour El Houda Souissi\n"
        "Email: nour.el.houda [at] outlook [dot] com\n"
        "Mobile: +216 5 4 1 4 2 3 1 6\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=[],
        min_confidence=0.6,
        use_semantic=False,
    )
    assert payload["extracted_email"] == "nour.el.houda@outlook.com"
    assert re.sub(r"\D", "", payload["extracted_phone"] or "") == "21654142316"


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


def test_heading_candidate_does_not_treat_skill_list_as_heading():
    is_heading, key, weight = cv_parser._heading_candidate("HTML5, PHP OOP, CSS, SQL, JavaScript")
    assert is_heading is False
    assert key == ""
    assert weight == 0.0


def test_parse_cv_safe_keeps_multiple_skills_from_comma_certification_line(monkeypatch):
    text = (
        "CERTIFICATIONS\n"
        "HTML5, PHP OOP, CSS, SQL, JavaScript, Symfony, Zend Framework\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=["sql", "css", "javascript", "php oop", "html"],
        min_confidence=0.6,
        use_semantic=False,
    )
    names = {str(r["skill"]).lower() for r in payload["extracted_skills"]}
    assert "sql" in names
    assert "css" in names
    assert "javascript" in names
    assert len(names) >= 3

    conf = {str(r["skill"]).lower(): float(r["confidence"]) for r in payload["extracted_skills"]}
    # Open-vocab section skills should not be crushed to near-zero confidence.
    if "symfony" in conf:
        assert conf["symfony"] >= 0.5


def test_parse_cv_safe_separates_certifications_projects_and_skills(monkeypatch):
    text = (
        "CERTIFICATIONS\n"
        "- AWS Certified Developer Associate\n"
        "PROJECTS\n"
        "- Built an internship matching platform using Angular and FastAPI\n"
        "SKILLS\n"
        "- Python\n"
        "- SQL\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=["python", "sql", "angular", "fastapi"],
        min_confidence=0.6,
        use_semantic=False,
    )

    assert "AWS Certified Developer Associate" in payload["certifications"]
    assert "Built an internship matching platform using Angular and FastAPI" in payload["hands_on_projects"]
    assert payload["extraction_channels"]["certification"]
    assert payload["extraction_channels"]["hands_on_project"]

    skill_set = {str(skill).lower() for skill in payload["skills"]}
    assert "python" in skill_set
    assert "sql" in skill_set
    assert "aws certified developer associate" not in skill_set
    assert "built an internship matching platform using angular and fastapi" not in skill_set
    assert payload["project_skill_links"]
    linked_skills = {str(item.get("skill", "")).lower() for item in payload["project_skill_links"]}
    assert "angular" in linked_skills
    assert "fastapi" in linked_skills
    assert "angular" in payload["extraction_channels"]["project_validated_skill"]
    assert "fastapi" in payload["extraction_channels"]["project_validated_skill"]


def test_parse_cv_safe_context_strength_penalizes_weak_claims(monkeypatch):
    text = (
        "SKILLS\n"
        "- Basic Python familiarity\n"
        "- SQL\n"
        "PROJECTS\n"
        "- Built analytics dashboard using SQL and Python\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=["python", "sql"],
        min_confidence=0.6,
        use_semantic=False,
    )
    by_skill = {
        str(row.get("skill", "")).lower(): row
        for row in payload["extracted_skills"]
        if isinstance(row, dict) and row.get("skill")
    }
    assert "python" in by_skill and "sql" in by_skill
    assert float(by_skill["python"].get("context_strength", 0.0)) <= float(by_skill["sql"].get("context_strength", 1.0))
    assert isinstance(payload["project_skill_links"], list)


def test_parse_cv_safe_project_links_include_evidence_span(monkeypatch):
    text = (
        "HANDS-ON PROJECTS\n"
        "- Implemented payment API using Python and FastAPI for e-commerce platform\n"
        "SKILLS\n"
        "- Python\n"
        "- FastAPI\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=["python", "fastapi"],
        min_confidence=0.6,
        use_semantic=False,
    )
    assert payload["project_skill_links"]
    link = payload["project_skill_links"][0]
    assert "evidence_span" in link
    assert isinstance(link["evidence_span"], str)
    assert 0.0 <= float(link.get("confidence", 0.0)) <= 1.0


def test_extract_sections_detects_hands_on_security_projects_heading():
    text = (
        "CERTIFICATIONS\n"
        "- eJPT - eLearnSecurity Junior Penetration Tester\n"
        "Hands-On Security Projects\n"
        "- Active Directory Penetration Testing Lab\n"
    )
    sections, _weights = _extract_sections(text)
    assert "certifications" in sections
    assert "hands-on security projects" in sections
    assert all("active directory penetration testing lab" not in line.lower() for line in sections["certifications"])


def test_parse_cv_safe_keeps_certifications_projects_and_skills_mutually_exclusive(monkeypatch):
    text = (
        "CERTIFICATIONS\n"
        "- eCPPT - eLearnSecurity Certified Professional Penetration Tester\n"
        "- eJPT - eLearnSecurity Junior Penetration Tester\n"
        "- Hack The Box Pro Labs: Puppet\n"
        "Hands-On Security Projects\n"
        "- Active Directory Penetration Testing Lab (Personal Project) Jan 2025\n"
        "- LDAP and SMB reconnaissance\n"
        "SKILLS\n"
        "- Bloodhound\n"
        "- Impacket\n"
        "- Tunisie\n"
        "- Frameworks\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=["bloodhound", "impacket", "ejpt", "ecppt"],
        min_confidence=0.6,
        use_semantic=False,
    )

    assert payload["certifications"] == [
        "eCPPT - eLearnSecurity Certified Professional Penetration Tester",
        "eJPT - eLearnSecurity Junior Penetration Tester",
        "Hack The Box Pro Labs: Puppet",
    ]
    assert "Active Directory Penetration Testing Lab (Personal Project) Jan 2025" in payload["hands_on_projects"]
    skill_set = {str(skill).lower() for skill in payload["skills"]}
    assert "bloodhound" in skill_set
    assert "impacket" in skill_set
    assert "ejpt" not in skill_set
    assert "ecppt" not in skill_set
    assert "hands-on security projects" not in skill_set
    assert "frameworks" not in skill_set
    assert "tunisie" not in skill_set


def test_parse_cv_safe_project_detail_bullets_stay_under_parent_project(monkeypatch):
    text = (
        "Hands-On Security Projects\n"
        "- Active Directory Penetration Testing Lab (Personal Project) Jan 2025\n"
        "- Active Directory enumeration, LDAP and SMB reconnaissance, BloodHound attack path analysis\n"
        "- Lateral movement, persistence techniques, IDS and IPS evasion\n"
        "SKILLS\n"
        "- Bloodhound\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=["bloodhound"],
        min_confidence=0.6,
        use_semantic=False,
    )

    assert payload["hands_on_projects"] == [
        "Active Directory Penetration Testing Lab (Personal Project) Jan 2025",
    ]


def test_parse_cv_safe_stops_certifications_at_technical_skills_boundary(monkeypatch):
    text = (
        "CERTIFICATIONS\n"
        "- eJPT - eLearnSecurity Junior Penetration Tester\n"
        "Technical Skills\n"
        "- Cisco SD WAN architecture, vManage and vSmart deployment\n"
        "- Offensive Security and Active Directory\n"
        "Hands-On Security Projects\n"
        "- Active Directory Penetration Testing Lab\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=[],
        min_confidence=0.6,
        use_semantic=False,
    )

    certs = {item.lower() for item in payload["certifications"]}
    assert "ejpt - elearnsecurity junior penetration tester" in certs
    assert not any("cisco sd wan architecture" in cert for cert in certs)
    assert not any("offensive security and active directory" in cert for cert in certs)


def test_parse_cv_safe_prunes_skill_when_also_classified_as_certification(monkeypatch):
    text = (
        "CERTIFICATIONS\n"
        "- eJPT - eLearnSecurity Junior Penetration Tester\n"
        "SKILLS\n"
        "- eJPT\n"
        "- Python\n"
    )
    monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, **_kwargs: text)
    payload = parse_cv_safe(
        file_bytes=b"%PDF-1.4\n",
        filename="cv.pdf",
        known_skills=["ejpt", "python"],
        min_confidence=0.6,
        use_semantic=False,
    )
    skill_set = {str(skill).lower() for skill in payload["skills"]}
    assert "python" in skill_set
    assert "ejpt" not in skill_set


def test_open_vocab_noise_rejects_including_phrases():
    assert cv_parser._open_vocab_looks_like_noise_sentence("including html5, php oop, javascript, css, sql")
    assert cv_parser._open_vocab_looks_like_noise_sentence("experience in web application development")


def test_open_vocab_display_normalizes_leaky_labels_and_punctuation():
    assert cv_parser._display_open_vocab_skill("programming languages javascript") == "javascript"
    assert cv_parser._display_open_vocab_skill("strong | javascript") == "javascript"
    assert cv_parser._display_open_vocab_skill("symfony.") == "symfony"
    assert cv_parser._display_open_vocab_skill("php framework certificate zend") == "zend"


def test_open_vocab_phrase_ok_filters_short_acronyms():
    # Very short acronyms like "O" should be filtered outside skill sections   
    assert not cv_parser._open_vocab_phrase_ok("O", language_section=False)
    # But 3-letter ones like "PHP" should pass
    assert cv_parser._open_vocab_phrase_ok("PHP", language_section=False)

def test_evidence_based_gating_drops_weak_skills():
    rows = [
        {"skill": "python", "source": "exact", "evidence": []},  # Strong source, keep
        {"skill": "php oop", "source": "cv_section:skills", "evidence": []},  # Weak evidence, drop
        {"skill": "javascript", "source": "cv_section:skills", "evidence": ["developed web apps"]},  # Good evidence, keep
    ]
    filtered = cv_parser._apply_evidence_based_gating(rows)
    skills = {r["skill"] for r in filtered}
    assert "python" in skills
    assert "php oop" not in skills
    assert "javascript" in skills


def test_five_letter_spaced_heading_collapses_like_tools():
    text = "T O O L S\n- SQL\n- Docker\n"
    sections, weights = _extract_sections(text)
    assert "tools" in sections
    assert "sql" not in sections and "docker" not in sections
    assert any("sql" in ln.lower() for ln in sections.get("tools", []))


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


def test_detect_skills_with_confidence_ignores_education_catalog_noise():
    text = "FORMATION\nBEP au lycee blaringhem a bethune\n"
    rows = detect_skills_with_confidence(
        text=text,
        known_skills=["au lycee blaringhem a bethune"],
        min_confidence=0.6,
        use_semantic=False,
    )
    assert rows == []


def test_detect_title_french_project_manager():
    text = "LUCAS LEBLANC\nCHARGE DE PROJETS TI\nMontreal, Canada"
    assert detect_title(text) == "CHARGE DE PROJETS TI"


def test_clean_extracted_text_skips_noise_lines():
    cleaned = cv_parser._clean_extracted_text(
        "https://example.com\nmodeles-de-cv.com\nCOPYRIGHT 2024\nValid line\n"
    )
    assert "valid line" in cleaned.lower()
    assert "http" not in cleaned.lower()
    assert "copyright" not in cleaned.lower()


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


def test_clean_evidence_item_strips_noise_markers():
    cleaned = cv_parser._clean_evidence_item("including html5, php oop, css | strong")
    assert cleaned == "html5, php oop, css"


def test_source_label_and_band_mapping():
    assert cv_parser._source_label_key("exact:certifications") == "exact"
    assert cv_parser._source_label_key("cv_section:skills") == "section"
    assert cv_parser._source_label_key("ner_span:hf_ner") == "ner"
    assert cv_parser._confidence_band_for_source("exact:skills", 0.82) == "high"
    assert cv_parser._confidence_band_for_source("cv_section:skills", 0.62) == "low"


