from app.ai import runtime
from app.ai.matcher import CandidateMatcher
from app.core.config import settings
import json


def test_model_info_endpoint(client, admin_auth, monkeypatch, tmp_path):
    model_path = tmp_path / "matcher.joblib"
    CandidateMatcher(use_semantic=False).save(model_path)

    metrics_path = tmp_path / "matcher_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "dataset_version": "demo_dataset_v1",
                "trained_at_utc": "2026-03-27T00:00:00Z",
                "roc_auc": 0.91,
                "f1": 0.84,
            }
        ),
    encoding="utf-8",
)
    runtime.load_matcher_artifact(model_path)
    monkeypatch.setattr(settings, "AI_MODEL_PATH", str(model_path), raising=False)

    r = client.get("/ai/model-info", headers=admin_auth)
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["artifact_exists"] is True
    assert body["model_path"] == str(model_path)
    assert "metadata" in body
    assert body["metadata"].get("model_name") == "candidate_matcher"
    assert body["dataset_version"] == "demo_dataset_v1"
    assert body["trained_at_utc"] == "2026-03-27T00:00:00Z"
    assert body["metrics"]["roc_auc"] == 0.91
    assert body["metrics"]["f1"] == 0.84

