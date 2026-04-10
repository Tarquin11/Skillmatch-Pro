import io
import logging
import re
import uuid
import zipfile
from datetime import date, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.api.auth import get_current_active_user
from app.api.utils import apply_list_query
from app.core.structured_log import (
    EVENT_CV_PARSE_FAILURE,
    REASON_PARSING_FAIL,
    log_structured_event,
)
from app.db.database import get_db
from app.models.Employee_skill import EmployeeSkill
from app.models.employee import Employee
from app.models.skill import Skill
from app.schemas.candidate import CandidateListItem, CandidateUploadRespose
from app.schemas.common import ErrorResponse
from app.schemas.listing import ListQuery
from app.services.cv_parser import parse_cv_safe

router = APIRouter(
    prefix="/candidates",
    tags=["Candidates"],
    dependencies=[Depends(get_current_active_user)],
)
logger = logging.getLogger(__name__)

_ALLOWED_MIMES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
_FILENAME_NOISE_TOKENS = {"cv", "resume", "candidate", "profile", "final", "v", "version"}
_CANDIDATE_SORT_FIELDS = {
    "id": Employee.id,
    "name": Employee.full_name,
    "email": Employee.email,
    "title": Employee.position,
    "uploaded_at": Employee.created_at,
}


def _sniff_cv_mime(file_bytes: bytes) -> str:
    if not file_bytes:
        return "application/octet-stream"
    if file_bytes.startswith(b"%PDF-"):
        return "application/pdf"
    if file_bytes[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                names = {n.lower() for n in zf.namelist()}
                if "[content_types].xml" in names and "word/document.xml" in names:
                    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        except Exception:
            pass
        return "application/octet-stream"
    return "application/octet-stream"


def _safe_float(value: object | None) -> float:
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _display_name_from_filename(filename: str) -> str:
    stem = Path(filename or "candidate").stem
    tokens = [tok for tok in re.split(r"[^A-Za-z0-9]+", stem) if tok]
    clean_tokens = [tok for tok in tokens if tok.lower() not in _FILENAME_NOISE_TOKENS]
    if not clean_tokens:
        return "Candidate Upload"
    return " ".join(tok.capitalize() for tok in clean_tokens[:4])


def _split_first_last_name(full_name: str) -> tuple[str, str]:
    tokens = [tok for tok in str(full_name or "").split() if tok]
    if not tokens:
        return "Candidate", "Upload"
    if len(tokens) == 1:
        return tokens[0], "Upload"
    return tokens[0], " ".join(tokens[1:])


def _predicted_hire_date(predicted_experience_years: float | None) -> date | None:
    years = max(0.0, _safe_float(predicted_experience_years))
    if years <= 0.0:
        return None
    days = int(round(years * 365.25))
    return date.today() - timedelta(days=days)


def _experience_years_from_hire_date(hire_date: date | None) -> float | None:
    if hire_date is None:
        return None
    days = max(0, (date.today() - hire_date).days)
    return round(days / 365.25, 2)


def _candidate_item_from_employee(employee: Employee) -> CandidateListItem:
    skills = sorted(
        {
            str(link.skill.name).strip()
            for link in (employee.skills or [])
            if getattr(getattr(link, "skill", None), "name", None)
        }
    )
    uploaded_at = None
    if getattr(employee, "created_at", None) is not None:
        uploaded_at = employee.created_at.isoformat()
    return CandidateListItem(
        id=int(employee.id),
        full_name=str(employee.full_name or "").strip() or "Candidate Upload",
        email=str(employee.email or "").strip(),
        predicted_title=(str(employee.position or "").strip() or None),
        predicted_experience_years=_experience_years_from_hire_date(getattr(employee, "hire_date", None)),
        skills=skills,
        uploaded_at=uploaded_at,
    )


def _persist_candidate_profile(
    *,
    db: Session,
    filename: str,
    parsed: dict,
    created_by: int | None,
) -> Employee:
    display_name = _display_name_from_filename(filename)
    first_name, last_name = _split_first_last_name(display_name)
    predicted_title = str(parsed.get("predicted_title") or "").strip() or "Candidate"

    employee = Employee(
        employee_number=f"CV-{uuid.uuid4().hex[:10].upper()}",
        first_name=first_name,
        last_name=last_name,
        full_name=display_name,
        email=f"cv_{uuid.uuid4().hex[:12]}@candidate.local",
        department="candidate pool",
        position=predicted_title,
        employment_status="candidate",
        recruitment_source="cv_upload",
        hire_date=_predicted_hire_date(_safe_float(parsed.get("predicted_experience_years"))),
        created_by=created_by,
    )
    db.add(employee)
    db.flush()

    seen_skills: set[str] = set()
    for raw in parsed.get("skills") or []:
        name = str(raw or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen_skills:
            continue
        seen_skills.add(key)

        skill = db.query(Skill).filter(func.lower(Skill.name) == key).first()
        if skill is None:
            skill = Skill(name=name)
            db.add(skill)
            db.flush()

        db.add(EmployeeSkill(employee_id=int(employee.id), skill_id=int(skill.id), level=3))

    db.commit()
    db.refresh(employee)
    return employee


@router.get("/", response_model=list[CandidateListItem])
def list_candidates(
    params: ListQuery = Depends(),
    db: Session = Depends(get_db),
):
    query = db.query(Employee).filter(func.lower(func.coalesce(Employee.recruitment_source, "")) == "cv_upload")
    if params.search:
        term = params.search.strip().lower()
        if term:
            query = query.filter(
                or_(
                    func.lower(func.trim(Employee.full_name)).like(f"%{term}%"),
                    func.lower(func.trim(Employee.email)).like(f"%{term}%"),
                    func.lower(func.trim(Employee.position)).like(f"%{term}%"),
                    func.lower(func.trim(Employee.employee_number)).like(f"%{term}%"),
                )
            )

    rows = apply_list_query(
        query,
        sort_by=params.sort_by,
        sort_dir=params.sort_dir,
        sort_map=_CANDIDATE_SORT_FIELDS,
        skip=params.skip,
        limit=params.limit,
    ).all()
    return [_candidate_item_from_employee(row) for row in rows]


@router.post(
    "/upload_cv",
    response_model=CandidateUploadRespose,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def upload_cv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    if not file.filename or not file.filename.lower().endswith((".pdf", ".docx")):
        log_structured_event(
            logger,
            level=logging.WARNING,
            event=EVENT_CV_PARSE_FAILURE,
            reason=REASON_PARSING_FAIL,
            stage="extension_validation",
            filename=file.filename,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "invalid_file_type",
                "message": "Only PDF and DOCX files are supported.",
            },
        )

    try:
        contents = await file.read()
        sniffed = _sniff_cv_mime(contents)
        if sniffed not in _ALLOWED_MIMES:
            log_structured_event(
                logger,
                level=logging.WARNING,
                event=EVENT_CV_PARSE_FAILURE,
                reason=REASON_PARSING_FAIL,
                stage="mime_sniffing",
                filename=file.filename,
                sniffed_mime=sniffed,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "invalid_file_type",
                    "message": "Only PDF and DOCX files are supported.",
                },
            )
        if sniffed == "application/pdf":
            safe_name = file.filename or "cv.pdf"
            if not safe_name.lower().endswith(".pdf"):
                safe_name = "cv.pdf"
        else:
            safe_name = file.filename or "cv.docx"
            if not safe_name.lower().endswith(".docx"):
                safe_name = "cv.docx"

        known_skills = [name for (name,) in db.query(Skill.name).all() if name]
        parsed = parse_cv_safe(
            file_bytes=contents,
            filename=safe_name,
            known_skills=known_skills,
            min_confidence=0.6,
            use_semantic=False,
            use_hf_ner=True,
            use_semantic_augment=True,
        )
        try:
            _persist_candidate_profile(
                db=db,
                filename=(file.filename or safe_name),
                parsed=parsed,
                created_by=(int(getattr(current_user, "id")) if getattr(current_user, "id", None) is not None else None),
            )
        except Exception:
            db.rollback()
            parsed["degraded"] = True
            warnings = list(parsed.get("warnings") or [])
            warnings.append("Candidate profile could not be saved automatically.")
            parsed["warnings"] = warnings
            logger.exception("candidate_autosave_failure filename=%s", file.filename)

        return CandidateUploadRespose(
            filename=file.filename,
            ok=parsed["ok"],
            degraded=parsed["degraded"],
            errors=parsed["errors"],
            warnings=parsed["warnings"],
            text_length=parsed["text_length"],
            skills=parsed["skills"],
            skills_grouped=parsed["skills_grouped"],
            skill_hierarchy=parsed["skill_hierarchy"],
            skill_graph=parsed["skill_graph"],
            extracted_languages=parsed["extracted_languages"],
            language_details=parsed["language_details"],
            extraction_channels=parsed["extraction_channels"],
            extracted_skills=parsed["extracted_skills"],
            preview=parsed["preview"],
            predicted_title=parsed["predicted_title"],
            predicted_experience_years=parsed["predicted_experience_years"],
        )
    except HTTPException:
        raise
    except Exception:
        log_structured_event(
            logger,
            level=logging.ERROR,
            event=EVENT_CV_PARSE_FAILURE,
            reason=REASON_PARSING_FAIL,
            stage="pipeline_exception",
            filename=file.filename,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "cv_processing_failed",
                "message": "Failed to process CV file.",
            },
        )
