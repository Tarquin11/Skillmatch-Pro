from fastapi import APIRouter, Depends 
from app.ai.runtime import get_model_info
from app.api.auth import get_current_active_user
from app.core.config import settings
from app.schemas.ai import AIModelInfoResponse

router = APIRouter(prefix="/ai", tags=["ai"], dependencies=[Depends(get_current_active_user)])

@router.get("/model-info", response_model=AIModelInfoResponse)
def model_info():
    info = get_model_info(settings.AI_MODEL_PATH)
    return AIModelInfoResponse(
        model_loaded=bool(info.get("model_loaded", False)),
        artifact_exists=bool(info.get("artifact_exists", False)),
        model_path=str(info.get("model_path",settings.AI_MODEL_PATH)),
        autoload_enabled=bool(settings.AI_MODEL_AUTOLOAD),
        runtime_source=str(info.get("runtime_source", "heuristic")),
        metadata=dict(info.get("metadata", {})),
        dataset_version=(str(info.get("dataset_version"))if info.get("dataset_version") is not None else None),
        trained_at_utc=(str(info.get("trained_at_utc"))if info.get("trained_at_utc") is not None else None),
        metrics=dict(info.get("metrics",{})),
    )