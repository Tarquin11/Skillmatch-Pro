from pydantic import BaseModel, Field
from typing import List

class JobMatchRequest(BaseModel):
    job_title: str
    required_skills: List[str] = []
    min_experience: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=200)

class MatchCandidateOut(BaseModel):
    employee_id: int
    full_name: str
    score: float

class JobMatchResponse(BaseModel):
    job_title: str
    required_skills: List[str]
    min_experience: int
    results: List[MatchCandidateOut] 