from math import ceil
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.api.auth import get_current_active_user
from app.db.database import get_db
from app.models.employee import Employee
from app.schemas.match import JobMatchRequest, JobMatchResponse, MatchCandidateOut
from app.services.model_inference import ModelInferenceService

router = APIRouter(prefix="/match", tags=["matching"], dependencies=[Depends(get_current_active_user)])
inference_service = ModelInferenceService()


def _sort_candidates(rows: list[MatchCandidateOut], sort_by: str, sort_direction: str) -> list[MatchCandidateOut]:
    reverse = sort_direction == "desc"

    def _value(row: MatchCandidateOut):
        if sort_by == "name":
            return (row.full_name or "").lower()
        if sort_by == "title":
            return ""
        if sort_by == "experience":
            return 0.0
        return float(row.predicted_fit_score)

    return sorted(rows, key=_value, reverse=reverse)


@router.post("/job", response_model=JobMatchResponse)
@router.post("/jobs", response_model=JobMatchResponse)
def rank_candidates(
    payload: JobMatchRequest,
    page: int = Query(default=1, ge=1),
    page_size: int | None = Query(default=None, ge=1, le=200),
    sort_by: str = Query(default="score", pattern="^(score|name|title|experience)$"),
    sort_direction: str = Query(default="desc", pattern="^(asc|desc)$"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    employees = db.query(Employee).all()
    routing_key = f"user:{getattr(current_user, 'id', 'na')}|{getattr(current_user, 'email', '')}"

    ranked_rows = inference_service.rank_candidates(
        job_title=payload.job_title,
        required_skills=payload.required_skills,
        min_experience=payload.min_experience,
        employees=employees,
        limit=payload.limit,
        routing_key=routing_key,
    )
    allowed_fields = MatchCandidateOut.model_fields.keys()
    ranked = [
        MatchCandidateOut(**{key: value for key, value in row.items() if key in allowed_fields})
        for row in ranked_rows
    ]

    sorted_ranked = _sort_candidates(ranked, sort_by=sort_by, sort_direction=sort_direction)

    effective_page_size = page_size if page_size is not None else int(payload.limit)
    total_results = len(sorted_ranked)
    total_pages = max(1, ceil(total_results / effective_page_size)) if total_results else 1

    start = (page - 1) * effective_page_size
    end = start + effective_page_size
    paged_results = sorted_ranked[start:end] if start < total_results else []

    return JobMatchResponse(
        job_title=payload.job_title,
        required_skills=payload.required_skills,
        min_experience=payload.min_experience,
        results=paged_results,
        total_results=total_results,
        page=page,
        page_size=effective_page_size,
        total_pages=total_pages,
        has_next=end < total_results,
        has_prev=page > 1 and total_results > 0,
        sort_by=sort_by,
        sort_direction=sort_direction,
    )

