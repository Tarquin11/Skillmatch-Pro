from app.services.training_recommendation import build_training_recommendations

def test_build_training_recommendations_detects_gaps():
    out = build_training_recommendations(
        job_title="data analyst",
        required_skills=["Python", "SQL", "PowerBI"],
        owned_skills=["Python"],
        top_k=2,
    )
    assert out["matched_skills"] == ["python"]
    assert out["missing_skills"] == ["sql", "powerbi"]
    assert out["skill_gap_ratio"] == 0.6667
    assert len(out["learning_recommendations"]) == 2


def test_build_training_recommendations_handles_empty_requirements():
    out = build_training_recommendations(
        job_title="Anything",
        required_skills=[],
        owned_skills=["python"],
    )
    assert out["skill_gap_ratio"] == 0.0
    assert out["learning_recommendations"] == []
