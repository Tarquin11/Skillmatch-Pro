from __future__ import annotations
from pathlib import Path
from typing import Optional , Any
import logging
from app.ai.matcher import CandidateMatcher
logger = logging.getLogger(__name__)
_matcher: Optional[CandidateMatcher] = None
_matcher_metadata: dict[str, Any] = {}
_loaded_model_path: Path | None = None
def get_matcher() -> Optional[CandidateMatcher]:
    return _matcher
def get_matcher_metadata() -> dict[str, Any]:
    return dict(_matcher_metadata)
def load_matcher_artifact(path: str | Path) -> Optional[CandidateMatcher]:
    global _matcher , _matcher_metadata , _loaded_model_path
    model_path = Path(path)
    if not model_path.exists():
        _matcher = None
        _matcher_metadata = {}
        _loaded_model_path = None
        return None
    try:
        _matcher = CandidateMatcher.load(model_path)
        _matcher_metadata = CandidateMatcher.read_artifact_metadata(model_path)
        _loaded_model_path = model_path
    except Exception:
        logger.exception("ai_model_load_failure path=%s", model_path)
        _matcher = None
        _matcher_metadata = {}
        _loaded_model_path = None
    return _matcher

def get_model_info(path: str | Path) -> dict[str, Any]:
    model_path = Path(path)
    artifact_exists = model_path.exists()

    loaded_for_requested_path = (
        _matcher is not None and _loaded_model_path is not None and _loaded_model_path == model_path
    )

    if loaded_for_requested_path:
        metadata = get_matcher_metadata()
        runtime_source = "model" if getattr(_matcher, "is_fitted", False) else "heuristic"
    else:
        metadata = CandidateMatcher.read_artifact_metadata(model_path) if artifact_exists else {}
        runtime_source = "heuristic"

    return {
        "model_loaded": bool(loaded_for_requested_path),
        "artifact_exists": bool(artifact_exists),
        "model_path": str(model_path),
        "runtime_source": runtime_source,
        "metadata": metadata or {},
    }
