import numpy as np

from app.scripts.evaluate_generalization import _scenario_metrics
from app.scripts.run_scheduled_holdout_eval import _resolve_threshold


def test_resolve_threshold_uses_policy_when_cli_missing():
    policy = {
        "generalization": {
            "classification_threshold": 0.45,
        }
    }
    assert _resolve_threshold(None, policy) == 0.45


def test_resolve_threshold_prefers_cli_value():
    policy = {
        "generalization": {
            "classification_threshold": 0.45,
        }
    }
    assert _resolve_threshold(0.5, policy) == 0.5


def test_scenario_metrics_falls_back_when_semantic_fails(monkeypatch):
    class FakeFeatureEngineer:
        def __init__(self):
            self.use_semantic = True
            self.calls = 0

        def vectorize_pairs(self, _pairs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("semantic backend unavailable")
            x = np.asarray([[0.0], [1.0]], dtype=np.float32)
            y = np.asarray([0, 1], dtype=np.int32)
            return x, y

    class FakeModel:
        @staticmethod
        def predict_proba(_x):
            return np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)

    class FakeMatcher:
        def __init__(self):
            self.feature_engineer = FakeFeatureEngineer()
            self.model = FakeModel()
            self.is_fitted = True

        @staticmethod
        def _classification_metrics(_y_true, _y_prob, threshold=0.5):
            return 1.0, 1.0, 1.0

        @staticmethod
        def _ranking_metrics(_y_true, _y_prob, query_ids, k=10):
            return {
                "precision_at_k": 1.0,
                "recall_at_k": 1.0,
                "map_at_k": 1.0,
                "ndcg_at_k": 1.0,
            }

    monkeypatch.setattr(
        "app.scripts.evaluate_generalization.preprocess_training_pairs",
        lambda rows: rows,
    )

    out = _scenario_metrics(
        FakeMatcher(),
        pairs=[{"query_id": "q1"}, {"query_id": "q1"}],
        k=10,
        threshold=0.45,
    )

    assert out["semantic_enabled"] is False
    assert "semantic_fallback" in out
    assert out["rows"] == 2

