from pydantic import Field
from typing import List , Dict
from app.schemas.common import StrictBaseModel

class JobMatchRequest(StrictBaseModel):
    job_title: str
    required_skills: List[str] = []
    min_experience: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=200)

class LearningRecommendationOut(StrictBaseModel):
    missing_skill: str
    learning_topic: str
    recommended_courses: List[str] = Field(default_factory=list)
    priority_score: float = Field(default=0.0, ge=0.0, le=1.0)

class MatchCandidateOut(StrictBaseModel):
    employee_id: int
    full_name: str
    score: float
    predicted_fit_score: float
    scoring_source: str
    feature_breakdown: Dict[str, float]
    top_reasons: List[str]
    matched_skills:List[str] = Field(default_factory=list)
    skill_gaps: List[str] = Field(default_factory=list)
    skill_gap_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    learning_recommendations: List[LearningRecommendationOut] = Field(default_factory=list)

class JobMatchResponse(StrictBaseModel):
    job_title: str
    required_skills: List[str]
    min_experience: int
    results: List[MatchCandidateOut] 
