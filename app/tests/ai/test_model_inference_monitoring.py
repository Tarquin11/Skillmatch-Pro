import logging
from datetime import date
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


def test_collapsed_model_scores_trigger_heuristic_recovery(monkeypatch):
    service = ModelInferenceService()
    service._COLLAPSED_SCORE_MIN_ROWS = 2

    class ConstantMatcher:
        is_fitted = True
        model = None

        def predict_scores(self, employees, _job_payload, batch_size=256):
            _ = batch_size
            return [
                {
                    "employee_id": int(emp.id),
                    "score": 0.0174,
                    "score_percent": 1.74,
                    "source": "model",
                    "features": {},
                }
                for emp in employees
            ]

    monkeypatch.setattr("app.services.model_inference.get_matcher", lambda: ConstantMatcher())

    employees = [_employee(1), _employee(2), _employee(3)]

    def fake_heuristic_row(*, employee, scoring_source, **_kwargs):
        score = 30.0 + (employee.id * 5.0)
        return {
            "employee_id": int(employee.id),
            "full_name": employee.full_name,
            "score": score,
            "predicted_fit_score": score,
            "score_raw": score,
            "predicted_fit_score_raw": score,
            "scoring_source": scoring_source,
            "feature_breakdown": {"title_alignment": 0.5},
            "top_reasons": ["Job title alignment (0.50)."],
            "matched_skills": [],
            "skill_gaps": [],
            "skill_gap_ratio": 1.0,
            "learning_recommendations": [],
        }

    monkeypatch.setattr(service, "_build_heuristic_row", fake_heuristic_row)

    rows = service.rank_candidates(
        job_title="web developer",
        required_skills=["html", "css"],
        min_experience=0,
        employees=employees,
        limit=10,
    )

    scores = [float(r["predicted_fit_score"]) for r in rows]
    assert len(set(scores)) > 1
    assert all("heuristic_recovery" in str(r.get("scoring_source", "")) for r in rows)


def test_required_skill_guardrail_caps_high_semantic_low_skill_scores(monkeypatch):
    service = ModelInferenceService()

    class SemanticHeavyMatcher:
        is_fitted = True
        model = None

        def predict_scores(self, employees, _job_payload, batch_size=256):
            _ = batch_size
            return [
                {
                    "employee_id": int(emp.id),
                    "score": 0.4039,
                    "score_percent": 40.39,
                    "source": "model+heuristic_blend",
                    "features": {
                        "skill_overlap": 0.0,
                        "missing_skill_ratio": 1.0,
                        "semantic_similarity": 0.54,
                    },
                }
                for emp in employees
            ]

    monkeypatch.setattr("app.services.model_inference.get_matcher", lambda: SemanticHeavyMatcher())
    monkeypatch.setattr(
        "app.services.model_inference.build_training_recommendations",
        lambda **_kwargs: {
            "matched_skills": [],
            "missing_skills": ["html", "css", "javascript"],
            "skill_gap_ratio": 1.0,
            "learning_recommendations": [],
        },
    )

    employees = [_employee(1), _employee(2)]
    rows = service.rank_candidates(
        job_title="web developer",
        required_skills=["html", "css", "javascript"],
        min_experience=1,
        employees=employees,
        limit=10,
    )

    assert rows
    assert all(float(r["predicted_fit_score"]) <= 15.0 for r in rows)


def test_relevance_filter_drops_zero_skill_and_irrelevant_title_rows(monkeypatch):
    service = ModelInferenceService()

    class MixedMatcher:
        is_fitted = True
        model = None

        def predict_scores(self, employees, _job_payload, batch_size=256):
            _ = batch_size
            out = []
            for emp in employees:
                if int(emp.id) == 1:
                    out.append(
                        {
                            "employee_id": 1,
                            "score": 0.1855,
                            "score_percent": 18.55,
                            "source": "model+heuristic_blend",
                            "features": {"skill_overlap": 0.34, "semantic_similarity": 0.52},
                        }
                    )
                else:
                    out.append(
                        {
                            "employee_id": int(emp.id),
                            "score": 0.4039,
                            "score_percent": 40.39,
                            "source": "model+heuristic_blend",
                            "features": {"skill_overlap": 0.0, "semantic_similarity": 0.54},
                        }
                    )
            return out

    monkeypatch.setattr("app.services.model_inference.get_matcher", lambda: MixedMatcher())

    emp_relevant = SimpleNamespace(
        id=1,
        full_name="Relevant Dev",
        first_name="Relevant",
        last_name="Dev",
        position="Backend Developer",
        skills=[SimpleNamespace(name="javascript")],
    )
    emp_irrelevant = SimpleNamespace(
        id=2,
        full_name="Irrelevant Accountant",
        first_name="Irrelevant",
        last_name="Accountant",
        position="Accountant",
        skills=[SimpleNamespace(name="excel")],
    )

    rows = service.rank_candidates(
        job_title="web developer",
        required_skills=["html", "css", "javascript"],
        min_experience=0,
        employees=[emp_relevant, emp_irrelevant],
        limit=10,
    )

    assert len(rows) == 1
    assert int(rows[0]["employee_id"]) == 1


def test_experience_guardrail_penalizes_under_minimum_experience(monkeypatch):
    service = ModelInferenceService()

    class StableMatcher:
        is_fitted = True
        model = None

        def predict_scores(self, employees, _job_payload, batch_size=256):
            _ = batch_size
            return [
                {
                    "employee_id": int(emp.id),
                    "score": 0.1919,
                    "score_percent": 19.19,
                    "source": "model+heuristic_blend",
                    "features": {"skill_overlap": 0.35, "semantic_similarity": 0.5},
                }
                for emp in employees
            ]

    monkeypatch.setattr("app.services.model_inference.get_matcher", lambda: StableMatcher())
    monkeypatch.setattr(
        "app.services.model_inference.build_training_recommendations",
        lambda **_kwargs: {
            "matched_skills": ["javascript"],
            "missing_skills": ["html", "css"],
            "skill_gap_ratio": 2.0 / 3.0,
            "learning_recommendations": [],
        },
    )

    candidate = SimpleNamespace(
        id=11,
        full_name="Junior Dev",
        first_name="Junior",
        last_name="Dev",
        position="Backend Developer",
        skills=[SimpleNamespace(name="javascript")],
        hire_date=None,  # zero inferred years
    )

    rows_min2 = service.rank_candidates(
        job_title="web developer",
        required_skills=["html", "css", "javascript"],
        min_experience=2,
        employees=[candidate],
        limit=10,
    )
    rows_min0 = service.rank_candidates(
        job_title="web developer",
        required_skills=["html", "css", "javascript"],
        min_experience=0,
        employees=[candidate],
        limit=10,
    )

    assert rows_min2 and rows_min0
    assert float(rows_min2[0]["predicted_fit_score"]) <= 10.0
    assert float(rows_min0[0]["predicted_fit_score"]) >= 19.0


def test_relevance_filter_drops_low_experience_when_minimum_is_set(monkeypatch):
    service = ModelInferenceService()

    class SimpleMatcher:
        is_fitted = True
        model = None

        def predict_scores(self, employees, _job_payload, batch_size=256):
            _ = batch_size
            return [
                {
                    "employee_id": int(emp.id),
                    "score": 0.1919,
                    "score_percent": 19.19,
                    "source": "model+heuristic_blend",
                    "features": {"skill_overlap": 0.35, "semantic_similarity": 0.5},
                }
                for emp in employees
            ]

    monkeypatch.setattr("app.services.model_inference.get_matcher", lambda: SimpleMatcher())

    junior = SimpleNamespace(
        id=1,
        full_name="Junior Candidate",
        first_name="Junior",
        last_name="Candidate",
        position="Backend Developer",
        hire_date=None,
        skills=[SimpleNamespace(name="javascript")],
    )
    senior = SimpleNamespace(
        id=2,
        full_name="Senior Candidate",
        first_name="Senior",
        last_name="Candidate",
        position="Backend Developer",
        hire_date=date.today().replace(year=date.today().year - 5),
        skills=[SimpleNamespace(name="javascript")],
    )

    rows = service.rank_candidates(
        job_title="web developer",
        required_skills=["html", "css", "javascript"],
        min_experience=2,
        employees=[junior, senior],
        limit=10,
    )

    kept_ids = {int(r["employee_id"]) for r in rows}
    assert 2 in kept_ids
    assert 1 not in kept_ids
