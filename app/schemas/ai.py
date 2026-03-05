from typing import Any, Dict
from pydantic import BaseModel, Field

class AIModelInfoResponse(BaseModel):
    model_loaded: bool
    artifact_exists: bool
    model_path: str
    autoload_enabled: bool
    runtime_source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
