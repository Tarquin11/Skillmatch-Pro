from pydantic import Field
from typing import List
from app.schemas.common import StrictBaseModel

class CandidateSkillExtraction(StrictBaseModel):
    skill: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str

class CandidateUploadRespose(StrictBaseModel):
    filename: str
    ok: bool = True
    degraded: bool = False
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    text_length: int = Field(default=0, ge=0)
    skills: List[str] = Field(default_factory=list)
    preview: str = ""
    extracted_skills: List[CandidateSkillExtraction] = Field(default_factory=list)
    predicted_title: str | None = None
    predicted_experience_years: float | None = None
