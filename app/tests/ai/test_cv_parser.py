from app.services.cv_parser import detect_skills, detect_skills_with_confidence


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
