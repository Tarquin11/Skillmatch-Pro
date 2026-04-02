from app.services.cv_parser import (
    detect_experience_years,
    detect_skills,
    detect_skills_with_confidence,
    detect_title,
    extract_text,
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
    assert extract_text(b"plain text bytes", "resume.txt") == ""


def test_detect_title_returns_none_for_empty_text():
    assert detect_title("") is None


def test_detect_skills_with_empty_known_skills_returns_empty():
    rows = detect_skills_with_confidence("Python SQL", known_skills=[])
    assert rows == []
