from typing import Optional
from pydantic import Field, ConfigDict
from app.schemas.common import StrictBaseModel

class SkillBase(StrictBaseModel):
    name: Optional[str] = None
    model_config = ConfigDict(from_attributes=True, extra="forbid")

class SkillCreate(SkillBase):
    name: str

class SkillUpdate(SkillBase):
    pass

class SkillOut(SkillBase):
    id: int

class EmployeeSkillAssignRequest(StrictBaseModel):
    skill_id: int
    level: int = Field(default=1, ge=1, le=5)  

class EmployeeSkillOut(StrictBaseModel):
    id: int
    employee_id: int
    skill_id: int
    level: Optional[int] = None
    model_config = ConfigDict(from_attributes=True, extra="forbid")


