from typing import Optional
from pydantic import BaseModel, ConfigDict

class DepartementBase(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)

class DepartementCreate(DepartementBase):
    name: str

class DepartementUpdate(DepartementBase):
    pass

class DepartementOut(DepartementBase):
    id: int