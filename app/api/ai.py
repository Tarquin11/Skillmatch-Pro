from fastapi import APIRouter, Depends 
from pathlib import Path
from app.ai.runtime import get_model_info
from app.api.auth import get_current_active_user
from app.core.config import settings
from app.schemas.ai import AIModelInfoResponse

router = APIRouter(prefix="/ai", tags=["ai"], dependencies=[Depends(get_current_active_user)])

@router.get("/model-info", response_model=AIModelInfoResponse)
def model_info():
    info = get_model_info(settings.AI_MODEL_PATH)
    canary_path = Path(settings.AI_CANARY_MODEL_PATH).expanduser()
    canary_exists = canary_path.exists()
    metadata = dict(info.get("metadata", {}))
    metadata["canary"] = {
        "enabled": bool(settings.AI_CANARY_ENABLED),
        "traffic_percent":float(settings.AI_CANARY_TRAFFIC_PERCENT),
        "model_path":str(canary_path),
        "model_exists": bool(canary_exists),
    }
    return AIModelInfoResponse(
        model_loaded=bool(info.get("model_loaded", False)),
        artifact_exists=bool(info.get("artifact_exists", False)),
        model_path=str(info.get("model_path",settings.AI_MODEL_PATH)),
        autoload_enabled=bool(settings.AI_MODEL_AUTOLOAD),
        runtime_source=str(info.get("runtime_source", "heuristic")),
        metadata=metadata,
        dataset_version=(str(info.get("dataset_version"))if info.get("dataset_version") is not None else None),
        trained_at_utc=(str(info.get("trained_at_utc"))if info.get("trained_at_utc") is not None else None),
        metrics=dict(info.get("metrics",{})),
        canary_enabled=bool(settings.AI_CANARY_ENABLED),
        canary_traffic_percent=float(settings.AI_CANARY_TRAFFIC_PERCENT),
        canary_model_path=str(canary_path),
        canary_model_exists=bool(canary_exists),
        promotion_gate_checked_at_utc=(str(info.get("promotion_gate_checked_at_utc")) if info.get("promotion_gate_checked_at_utc") is not None else None),
        promotion_gate_passed=(bool(info.get("promotion_gate_passed")) if isinstance(info.get("promotion_gate_passed"), bool) else None),
        metric_gate_passed=(bool(info.get("metric_gate_passed")) if isinstance(info.get("metric_gate_passed"), bool) else None),
        drift_gate_passed=(bool(info.get("drift_gate_passed")) if isinstance(info.get("drift_gate_passed"), bool) else None),
        generalization_gate_passed=(bool(info.get("generalization_gate_passed")) if isinstance(info.get("generalization_gate_passed"), bool) else None),
        robustness_gate_passed=(bool(info.get("robustness_gate_passed")) if isinstance(info.get("robustness_gate_passed"), bool) else None),
        last_generalization_eval_at_utc=(str(info.get("last_generalization_eval_at_utc")) if info.get("last_generalization_eval_at_utc") is not None else None),
        last_generalization_scheduled_eval_at_utc=(str(info.get("last_generalization_scheduled_eval_at_utc")) if info.get("last_generalization_scheduled_eval_at_utc") is not None else None),
        last_generalization_gate_checked_at_utc=(str(info.get("last_generalization_gate_checked_at_utc")) if info.get("last_generalization_gate_checked_at_utc") is not None else None),
        last_generalization_gate_passed=(bool(info.get("last_generalization_gate_passed")) if isinstance(info.get("last_generalization_gate_passed"), bool) else None),
    )
