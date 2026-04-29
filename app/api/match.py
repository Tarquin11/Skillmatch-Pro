from math import ceil
from datetime import date
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.api.auth import get_current_active_user
from app.db.database import get_db
from app.models.employee import Employee
from app.schemas.match import JobMatchRequest, JobMatchResponse, MatchCandidateOut
from app.services.model_inference import ModelInferenceService
from app.services.matching import calculate_weighted_score
from app.ai.preprocessing import normalize_performance, normalize_skill_name

router = APIRouter(prefix="/match", tags=["matching"], dependencies=[Depends(get_current_active_user)])
inference_service = ModelInferenceService()


def _sort_candidates(rows: list[MatchCandidateOut], sort_by: str, sort_direction: str) -> list[MatchCandidateOut]:
    reverse = sort_direction == "desc"

    def _value(row: MatchCandidateOut):
        if sort_by == "name":
            return (row.full_name or "").lower()
        if sort_by == "title":
            return (row.predicted_title or "").lower()
        if sort_by == "experience":
            return float(row.predicted_experience_years or 0.0)
        return float(row.predicted_fit_score)

    return sorted(rows, key=_value, reverse=reverse)


def _recover_flat_scores_if_needed(
    *,
    rows: list[dict],
    employees: list[Employee],
    payload: JobMatchRequest,
) -> list[dict]:
    if len(rows) < 2:
        return rows

    scores = [float(row.get("predicted_fit_score", 0.0)) for row in rows]
    if len({round(score, 2) for score in scores}) > 1:
        return rows

    by_id = {int(emp.id): emp for emp in employees if getattr(emp, "id", None) is not None}
    recovered: list[dict] = []
    for row in rows:
        employee_id = row.get("employee_id")
        employee = by_id.get(int(employee_id)) if employee_id is not None else None
        if employee is None:
            recovered.append(row)
            continue

        h = calculate_weighted_score(
            employee=employee,
            job_title=payload.job_title,
            required_skills=payload.required_skills,
            min_experience=payload.min_experience,
            use_semantic=False,
        )
        experience_years = _employee_experience_years(employee)
        experience_norm = min(1.0, max(0.0, experience_years / 20.0))
        title_alignment = _title_alignment(payload.job_title, getattr(employee, "position", "") or "")
        skill_coverage = max(0.0, min(1.0, 1.0 - float(row.get("skill_gap_ratio", 1.0))))
        performance = normalize_performance(getattr(employee, "performance_score", None))
        recovery_signal = 100.0 * (
            (0.35 * skill_coverage)
            + (0.25 * title_alignment)
            + (0.25 * experience_norm)
            + (0.15 * performance)
        )

        model_score = float(row.get("predicted_fit_score", 0.0))
        heuristic_score = float(h.get("total", 0.0))
        blended = round((0.15 * model_score) + (0.55 * heuristic_score) + (0.30 * recovery_signal), 2)

        patched = dict(row)
        patched["score"] = blended
        patched["predicted_fit_score"] = blended
        patched["score_raw"] = float(blended)
        patched["predicted_fit_score_raw"] = float(blended)
        patched["predicted_title"] = (str(getattr(employee, "position", "")).strip() or None)
        patched["predicted_experience_years"] = round(experience_years, 2)
        patched["scoring_source"] = f"{row.get('scoring_source', 'model')}+api_recovery"
        recovered.append(patched)

    return recovered


def _employee_experience_years(employee: Employee) -> float:
    hire_date = getattr(employee, "hire_date", None)
    if hire_date is None:
        return 0.0
    days = max(0, (date.today() - hire_date).days)
    return days / 365.25


def _title_alignment(job_title: str, position: str) -> float:
    job_tokens = {tok for tok in normalize_skill_name(job_title).split() if tok}
    pos_tokens = {tok for tok in normalize_skill_name(position).split() if tok}
    if not job_tokens or not pos_tokens:
        return 0.0
    return len(job_tokens & pos_tokens) / len(job_tokens)


@router.post("/job", response_model=JobMatchResponse)
def rank_candidates(
    payload: JobMatchRequest,
    page: int = Query(default=1, ge=1),
    page_size: int | None = Query(default=None, ge=1, le=200),
    sort_by: str = Query(default="score", pattern="^(score|name|title|experience)$"),
    sort_direction: str = Query(default="desc", pattern="^(asc|desc)$"),
    candidate_scope: str = Query(default="all", pattern="^(all|candidates)$"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    query = db.query(Employee)
    if candidate_scope == "candidates":
        query = query.filter(func.lower(func.coalesce(Employee.recruitment_source, "")) == "cv_upload")
    employees = query.all()
    routing_key = f"user:{getattr(current_user, 'id', 'na')}|{getattr(current_user, 'email', '')}"

    ranked_rows = inference_service.rank_candidates(
        job_title=payload.job_title,
        required_skills=payload.required_skills,
        min_experience=payload.min_experience,
        employees=employees,
        limit=payload.limit,
        routing_key=routing_key,
    )
    ranked_rows = _recover_flat_scores_if_needed(rows=ranked_rows, employees=employees, payload=payload)
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
