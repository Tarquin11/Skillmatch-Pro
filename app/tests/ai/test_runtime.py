from app.ai.matcher import CandidateMatcher
from app.ai.runtime import get_model_info, load_matcher_artifact
import json


def test_get_model_info_reads_metadata(tmp_path):
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
    loaded = load_matcher_artifact(model_path)
    assert loaded is not None
    

    info = get_model_info(model_path)
    assert info["artifact_exists"] is True
    assert info["model_loaded"] is True
    assert info["metadata"]["model_name"] == "candidate_matcher"
    assert info["dataset_version"] == "demo_dataset_v1"
    assert info["trained_at_utc"] == "2026-03-27T00:00:00Z"
    assert info["metrics"]["roc_auc"] == 0.91
    assert info["metrics"]["f1"] == 0.84

def test_get_model_info_missing_artifact(tmp_path):
    missing = tmp_path / "missing.joblib"
    load_matcher_artifact(missing)
    info = get_model_info(missing)
    assert info["artifact_exists"] is False
    assert info["model_loaded"] is False
    assert info["dataset_version"] is None
    assert info["trained_at_utc"] is None
    assert info["metrics"] == {}

