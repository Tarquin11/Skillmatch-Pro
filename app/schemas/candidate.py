from pydantic import BaseModel, Field
from typing import List

class CandidateSkillExtraction(BaseModel):
    skill: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str

class CandidateUploadRespose(BaseModel):
    filename: str
    skills: List[str]
    preview: str
    extracted_skills: List[CandidateSkillExtraction] = Field(default_factory=list)

