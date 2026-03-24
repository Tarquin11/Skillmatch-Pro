from typing import Optional
from pydantic import Field, ConfigDict
from app.schemas.common import StrictBaseModel

class jobBase(StrictBaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    department: Optional[str] = Field(default=None, alias="departement")
    model_config = ConfigDict(from_attributes=True, populate_by_name=True, extra="forbid")

class JobCreate(jobBase):
    title: str

class JobUpdate(jobBase):
    pass

class JobOut(jobBase):
    id: int

class JobSkillRequirementRequest(StrictBaseModel):
    skill_id: int
    required_level: int = Field(default=3, ge=1, le=5)
    weight: float = Field(default=1.0, gt=0)

class JobSkillRequirementOut(StrictBaseModel):
    id: int
    job_id: int
    skill_id: int
    required_level: Optional[int] = None
    weight: Optional[float] = None
    model_config = ConfigDict(from_attributes=True, extra="forbid")
