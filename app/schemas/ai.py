from typing import Any, Dict
from pydantic import Field
from app.schemas.common import StrictBaseModel

class AIModelInfoResponse(StrictBaseModel):
    model_loaded: bool
    artifact_exists: bool
    model_path: str
    autoload_enabled: bool
    runtime_source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dataset_version: str | None = None
    trained_at_utc: str | None = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    canary_enabled: bool = False
    canary_traffic_percent: float = 0.0
    canary_model_path: str | None = None
    canary_model_exists: bool = False