from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.api.auth import get_current_active_user
from app.db.database import get_db
from app.models.employee import Employee
from app.schemas.match import JobMatchRequest, JobMatchResponse, MatchCandidateOut
from app.services.matching import calculate_match_score
from app.ai.runtime import get_matcher

router = APIRouter(prefix="/match",tags=["matching"],dependencies=[Depends(get_current_active_user)],)

@router.post("/job", response_model=JobMatchResponse)
@router.post("/jobs", response_model=JobMatchResponse)
def rank_candidates(payload: JobMatchRequest, db: Session = Depends(get_db)):
    employees = db.query(Employee).all()
    ranked: list[MatchCandidateOut] = []
    matcher = get_matcher()
    if matcher and matcher.is_fitted:
        job_payload = {
            "title": payload.job_title,
            "required_skills": payload.required_skills,
            "min_experience": payload.min_experience,
        }
        by_id = {e.id: e for e in employees}
        model_ranked = matcher.rank_candidates(job_payload, employees, top_k=payload.limit)

        for row in model_ranked:
            emp_id = row.get("employee_id")
            emp = by_id.get(emp_id)
            full_name = (emp.full_name if emp else "Unknown")
            ranked.append(
                MatchCandidateOut(
                    employee_id=emp_id,
                    full_name=full_name,
                    score=round(float(row["score_percent"]), 2),
                )
            )
    else:
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
        ranked = ranked[: payload.limit]

    return JobMatchResponse(
        job_title=payload.job_title,
        required_skills=payload.required_skills,
        min_experience=payload.min_experience,
        results=ranked,
    )
