from app.ai.matcher import CandidateMatcher
from app.ai.runtime import get_model_info, load_matcher_artifact


def test_get_model_info_reads_metadata(tmp_path):
    model_path = tmp_path / "matcher.joblib"
    CandidateMatcher(use_semantic=False).save(model_path)

    loaded = load_matcher_artifact(model_path)
    assert loaded is not None

    info = get_model_info(model_path)
    assert info["artifact_exists"] is True
    assert info["model_loaded"] is True
    assert info["metadata"]["model_name"] == "candidate_matcher"


def test_get_model_info_missing_artifact(tmp_path):
    missing = tmp_path / "missing.joblib"
    load_matcher_artifact(missing)
    info = get_model_info(missing)
    assert info["artifact_exists"] is False
    assert info["model_loaded"] is False
