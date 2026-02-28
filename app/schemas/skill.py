from typing import Optional
from pydantic import BaseModel, Field

class SkillBase(BaseModel):
    name : Optional[str] = None

    class Config: 
        from_attributes = True

class SkillCreate(SkillBase):
    name : str

class SkillUpdate(SkillBase):
    pass

class SkillOut(SkillBase):
    id: int

class EmployeeSkillAssignRequest(BaseModel):
    skill_id: int
    level: int = Field(default=1, ge=1, le=5)  

class EmployeeSkillOut(BaseModel):
    id: int
    employee_id: int
    skill_id: int
    level: Optional[int] = None

    class Config:
        from_attributes = True


