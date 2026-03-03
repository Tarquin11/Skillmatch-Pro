from pydantic import BaseModel, Field
from typing import List , Dict

class JobMatchRequest(BaseModel):
    job_title: str
    required_skills: List[str] = []
    min_experience: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=200)

class MatchCandidateOut(BaseModel):
    employee_id: int
    full_name: str
    score: float
    predicted_fit_score: float
    scoring_source: str
    feature_breakdown: Dict[str, float]
    top_reasons: List[str]

class JobMatchResponse(BaseModel):
    job_title: str
    required_skills: List[str]
    min_experience: int
    results: List[MatchCandidateOut] 