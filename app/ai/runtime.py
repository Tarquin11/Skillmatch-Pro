from __future__ import annotations
from pathlib import Path
from typing import Optional , Any
import logging
import json
from app.ai.matcher import CandidateMatcher

logger = logging.getLogger(__name__)
_matcher: Optional[CandidateMatcher] = None
_matcher_metadata: dict[str, Any] = {}
_loaded_model_path: Path | None = None

_METRIC_FIELDS = (
    "train_size",
    "valid_size",
    "roc_auc",
    "average_precision",
    "precision",
    "recall",
    "f1",
    "precision_at_k",
    "recall_at_k",
    "map_at_k",
    "ndcg_at_k",
)


def _candidate_metrics_paths(model_path: Path) -> list[Path]:
    candidates: list[Path] = []
    stem = model_path.stem
    if stem.startswith("matcher_"):
        suffix = stem[len("matcher_") :].strip()
        if suffix:
            candidates.append(model_path.with_name(f"matcher_metrics_{suffix}.json"))
    candidates.append(model_path.with_name("matcher_metrics.json"))
    return candidates


def _load_metrics_payload(model_path: Path) -> dict[str, Any]:
    for metrics_path in _candidate_metrics_paths(model_path):
        if not metrics_path.exists():
            continue
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("ai_model_metrics_read_failure path=%s", metrics_path)
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _extract_metrics(payload: dict[str, Any]) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for key in _METRIC_FIELDS:
        if key not in payload:
            continue
        raw = payload.get(key)
        if raw is None:
            metrics[key] = None
            continue
        if isinstance(raw, bool):
            continue
        try:
            metrics[key] = float(raw)
        except (TypeError, ValueError):
            continue
    return metrics


def _enrich_model_info(model_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    metrics_payload = _load_metrics_payload(model_path)

    dataset_version = metrics_payload.get("dataset_version") or metadata.get("dataset_version")
    trained_at_utc = metrics_payload.get("trained_at_utc") or metadata.get("trained_at_utc")

    return {
        "dataset_version": str(dataset_version) if dataset_version is not None else None,
        "trained_at_utc": str(trained_at_utc) if trained_at_utc is not None else None,
        "metrics": _extract_metrics(metrics_payload),
    }

def _normalize_model_path(path: str | Path) -> Path:
    if isinstance(path, Path):
        candidate = path
    elif isinstance(path, str):
        raw = path.strip()
        if not raw:
            raise ValueError("AI model is empty")
        candidate = Path(raw)
    else: 
        raise TypeError(f"Invalid model path type : {type(path).__name__}")
    return candidate.expanduser().resolve(strict=False)

def get_matcher() -> Optional[CandidateMatcher]:
    return _matcher

def get_matcher_metadata() -> dict[str, Any]:
    return dict(_matcher_metadata)

def load_matcher_artifact(path: str | Path) -> Optional[CandidateMatcher]:
    global _matcher , _matcher_metadata , _loaded_model_path
    try:
        model_path = _normalize_model_path(path)
    except Exception:
        logger.exception("ai_model_invalid_path value=%r", path)
        _matcher = None
        _matcher_metadata = {}
        _loaded_model_path = None
        return None
    
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
    try:
        model_path = _normalize_model_path(path)
    except Exception:
        logger.exception("ai_model_invalid_path value=%r", path)
        return{
            "model_loaded" : False,
            "artifact_exists": False,
            "model_path": str(path),
            "runtime_source": "heuristic",
            "dataset_version": None,
            "trained_at_utc": None,
            "metrics":{},
            "metadata": {},
        }

    artifact_exists = model_path.exists()
    loaded_for_requested_path = (
        _matcher is not None 
        and _loaded_model_path is not None 
        and _loaded_model_path == model_path
    )

    if loaded_for_requested_path:
        metadata = get_matcher_metadata()
        runtime_source = "model" if getattr(_matcher, "is_fitted", False) else "heuristic"
    else:
        metadata = CandidateMatcher.read_artifact_metadata(model_path) if artifact_exists else {}
        runtime_source = "heuristic"

    enriched = _enrich_model_info(model_path, metadata or {}) if artifact_exists else {
        "dataset_version": None,
        "trained_at_utc": None,
        "metrics":{},
    }

    return {
        "dataset_version": enriched["dataset_version"],
        "trained_at_utc":enriched["trained_at_utc"],
        "metrics": enriched["metrics"],
        "model_loaded": bool(loaded_for_requested_path),
        "artifact_exists": bool(artifact_exists),
        "model_path": str(model_path),
        "runtime_source": runtime_source,
        "metadata": metadata or {},
    }
