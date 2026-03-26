import logging
from types import SimpleNamespace

import pytest

from app.services.model_inference import ModelInferenceService


def _employee(emp_id: int):
    return SimpleNamespace(
        id=emp_id,
        full_name=f"Emp {emp_id}",
        first_name="Emp",
        last_name=str(emp_id),
        skills=[],
    )


def test_logs_prediction_distribution(monkeypatch, caplog):
    service = ModelInferenceService()
    employees = [_employee(1), _employee(2), _employee(3)]

    monkeypatch.setattr("app.services.model_inference.get_matcher", lambda: None)

    def fake_weighted_score(*, employee, **_kwargs):
        return {
            "total": float(employee.id * 25),
            "skill_overlap": 0.5,
            "experience_score": 0.4,
            "semantic_similarity": 0.0,
            "performance_score": 0.7,
        }

    monkeypatch.setattr("app.services.model_inference.calculate_weighted_score", fake_weighted_score)

    caplog.set_level(logging.INFO)
    rows = service.rank_candidates(
        job_title="Data Analyst",
        required_skills=["python", "sql"],
        min_experience=1,
        employees=employees,
        limit=10,
    )

    assert len(rows) == 3
    assert any("ai_prediction_distribution" in rec.message for rec in caplog.records)


def test_logs_prediction_failures(monkeypatch, caplog):
    service = ModelInferenceService()
    employees = [_employee(1), _employee(2)]

    monkeypatch.setattr("app.services.model_inference.get_matcher", lambda: None)

    def flaky_weighted_score(*, employee, **_kwargs):
        if employee.id == 2:
            raise RuntimeError("boom")
        return {
            "total": 55.0,
            "skill_overlap": 0.5,
            "experience_score": 0.4,
            "semantic_similarity": 0.0,
            "performance_score": 0.7,
        }

    monkeypatch.setattr("app.services.model_inference.calculate_weighted_score", flaky_weighted_score)

    caplog.set_level(logging.ERROR)
    rows = service.rank_candidates(
        job_title="Data Analyst",
        required_skills=["python"],
        min_experience=1,
        employees=employees,
        limit=10,
    )

    assert len(rows) == 1
    assert any("ai_prediction_failure" in rec.message for rec in caplog.records)


def test_monitoring_frame_prefers_raw_scores():
    service = ModelInferenceService()
    rows = [
        {
            "predicted_fit_score": 100.0,
            "predicted_fit_score_raw": 83.4567,
            "scoring_source": "model",
        }
    ]

    df = service._build_monitoring_frame(rows)
    assert len(df) == 1
    # raw score should be used (83.4567% -> 0.834567), not rounded 100% score
    assert float(df.iloc[0]["predicted_score"]) == pytest.approx(0.834567, abs=1e-9)
