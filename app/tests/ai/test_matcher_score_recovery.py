import numpy as np
import pytest

from app.ai.matcher import CandidateMatcher


class _ConstantModel:
    def predict_proba(self, x):
        n = x.shape[0]
        pos = np.full((n, 1), 0.0174, dtype=np.float32)
        neg = 1.0 - pos
        return np.concatenate([neg, pos], axis=1)


class _LinearModel:
    def predict_proba(self, x):
        n = x.shape[0]
        # Vary output by first feature to avoid collapsed-score branch.
        pos = np.clip(0.08 + (0.2 * x[:, 0]), 0.0, 1.0).reshape(n, 1)
        neg = 1.0 - pos
        return np.concatenate([neg, pos], axis=1)


def test_predict_scores_recovers_from_collapsed_model_outputs(monkeypatch):
    matcher = CandidateMatcher(use_semantic=False)
    matcher.is_fitted = True
    matcher.model = _ConstantModel()

    def fake_features(employee_raw, _job_raw):
        emp_id = int(employee_raw["id"])
        return {
            "skill_overlap": 0.0,
            "missing_skill_ratio": 1.0,
            "experience_gap": 0.0,
            "experience_surplus": float(emp_id),
            "semantic_similarity": 0.0,
            "performance_score": 0.0,
            "engagement_score": 0.0,
            "satisfaction_score": 0.0,
            "tenure_years": 0.0,
            "currently_active": 1.0,
        }

    monkeypatch.setattr(matcher.feature_engineer, "create_features", fake_features)

    employees = [{"id": idx} for idx in range(1, 21)]
    results = matcher.predict_scores(
        employee_list=employees,
        job_raw={"title": "web developer", "required_skills": ["html", "css"], "min_experience": 0},
        batch_size=64,
    )

    assert results
    scores = [float(row["score_percent"]) for row in results]
    sources = {str(row["source"]) for row in results}

    assert len(set(scores)) > 1
    assert "model+heuristic_recovery" in sources


def test_predict_scores_blends_model_and_heuristic_when_not_collapsed(monkeypatch):
    matcher = CandidateMatcher(use_semantic=False)
    matcher.is_fitted = True
    matcher.model = _LinearModel()

    def fake_features(employee_raw, _job_raw):
        emp_id = int(employee_raw["id"])
        overlap = min(1.0, emp_id / 10.0)
        return {
            "skill_overlap": float(overlap),
            "missing_skill_ratio": float(1.0 - overlap),
            "experience_gap": 0.0,
            "experience_surplus": 0.0,
            "semantic_similarity": 0.0,
            "performance_score": 0.0,
            "engagement_score": 0.0,
            "satisfaction_score": 0.0,
            "tenure_years": 0.0,
            "currently_active": 1.0,
        }

    monkeypatch.setattr(matcher.feature_engineer, "create_features", fake_features)

    employees = [{"id": idx} for idx in range(1, 11)]
    results = matcher.predict_scores(
        employee_list=employees,
        job_raw={"title": "web developer", "required_skills": ["html", "css"], "min_experience": 0},
        batch_size=64,
    )

    assert results
    assert all(str(row["source"]) == "model+heuristic_blend" for row in results)

    # For first employee:
    # model = 0.08 + 0.2*0.1 = 0.10
    # heuristic = 0.35*0.1 + 0.10 = 0.135
    # default blend = 0.2*model + 0.8*heuristic = 0.128
    assert float(results[0]["score"]) == pytest.approx(0.128, abs=1e-6)
