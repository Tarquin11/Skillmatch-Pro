from typing import Optional
from pydantic import BaseModel, Field

class jobBase(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    department: Optional[str] = Field(default=None, alias="departement")

    class Config:
        from_attributes = True
        populate_by_name = True

class JobCreate(jobBase):
    title: str

class JobUpdate(jobBase):
    pass

class JobOut(jobBase):
    id: int

class JobSkillRequirementRequest(BaseModel):
    skill_id: int
    required_level: int = Field(default=3, ge=1, le=5)
    weight: float = Field(default=1.0, gt=0)

class JobSkillRequirementOut(BaseModel):
    id: int
    job_id: int
    skill_id: int
    required_level: Optional[int] = None
    weight: Optional[float] = None

    class Config:
        from_attributes = True