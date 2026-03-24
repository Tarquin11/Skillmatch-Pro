from pydantic import BaseModel, ConfigDict
from typing import Any, Optional

class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class ErrorDetail(BaseModel):
    code : str
    message : str

class ErrorResponse(BaseModel):
    detail : ErrorDetail
    errors: Optional[list[Any]] = None