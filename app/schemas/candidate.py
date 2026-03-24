from pydantic import Field
from typing import List
from app.schemas.common import StrictBaseModel

class CandidateSkillExtraction(StrictBaseModel):
    skill: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str

class CandidateUploadRespose(StrictBaseModel):
    filename: str
    skills: List[str]
    preview: str
    extracted_skills: List[CandidateSkillExtraction] = Field(default_factory=list)

