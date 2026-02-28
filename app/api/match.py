from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.api.auth import get_current_active_user
from app.db.database import get_db
from app.models.employee import Employee
from app.schemas.match import JobMatchRequest, JobMatchResponse, MatchCandidateOut
from app.services.matching import calculate_match_score

router = APIRouter(prefix="/match",tags=["matching"],dependencies=[Depends(get_current_active_user)],)

@router.post("/jobs", response_model=JobMatchResponse)
def rank_candidates(payload: JobMatchRequest, db: Session = Depends(get_db)):
    employees = db.query(Employee).all()
    ranked: list[MatchCandidateOut] = []

    for employee in employees:
        score = calculate_match_score(
            employee=employee,
            target_job=payload.job_title,
            required_skills=payload.required_skills,
            min_experience=payload.min_experience,
        )
        ranked.append(
            MatchCandidateOut(
                employee_id=employee.id,
                full_name=employee.full_name,
                score=round(float(score), 2),
            )
        )

    ranked.sort(key=lambda x: x.score, reverse=True)
    limited = ranked[: payload.limit]

    return JobMatchResponse(
        job_title=payload.job_title,
        required_skills=payload.required_skills,
        min_experience=payload.min_experience,
        results=limited,
    )
