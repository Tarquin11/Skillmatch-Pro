from typing import Optional
from pydantic import ConfigDict
from app.schemas.common import StrictBaseModel

class DepartementBase(StrictBaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    model_config = ConfigDict(from_attributes=True, extra="forbid")

class DepartementCreate(DepartementBase):
    name: str

class DepartementUpdate(DepartementBase):
    pass

class DepartementOut(DepartementBase):
    id: int
