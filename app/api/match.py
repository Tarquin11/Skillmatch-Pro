from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.api.auth import get_current_active_user
from app.db.database import get_db
from app.models.employee import Employee
from app.schemas.match import JobMatchRequest, JobMatchResponse, MatchCandidateOut
from app.services.model_inference import ModelInferenceService
router = APIRouter(prefix="/match",tags=["matching"],dependencies=[Depends(get_current_active_user)],)
inference_service = ModelInferenceService()
@router.post("/job", response_model=JobMatchResponse)
@router.post("/jobs", response_model=JobMatchResponse)
def rank_candidates(
    payload: JobMatchRequest, 
    db: Session = Depends(get_db)
):
    employees = db.query(Employee).all()
    ranked_rows = inference_service.rank_candidates(
        job_title=payload.job_title,
        required_skills=payload.required_skills,
        min_experience=payload.min_experience,
        employees=employees,
        limit=payload.limit,
    )
    ranked = [MatchCandidateOut(**row) for row in ranked_rows]
    return JobMatchResponse(
        job_title=payload.job_title,
        required_skills=payload.required_skills,
        min_experience=payload.min_experience,
        results=ranked,
    )