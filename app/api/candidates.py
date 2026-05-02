import io
import json
import logging
import re
import uuid
import zipfile
from datetime import date, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
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
from app.schemas.candidate import CandidateListItem, CandidateUpdateRequest, CandidateUploadRespose
from app.schemas.common import ErrorResponse
from app.schemas.listing import ListQuery
from app.services.active_learning import (
    get_resolved_unknown_entity_for_term,
    queue_certs_and_projects_for_review,
    queue_unknown_entities_for_parsed_cv,
    resolve_approved_skill_for_term,
    should_queue_skill_review,
)
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
_PARSED_EMAIL_RE = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.IGNORECASE)
_PREVIEW_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}\b")
_PREVIEW_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().\-]{6,}\d)(?!\w)")
_OBFUSCATED_AT_RE = re.compile(
    r"(?i)(?<=[A-Z0-9._%+\-])\s*(?:\[\s*at\s*\]|\(\s*at\s*\)|\bat\b)\s*(?=[A-Z0-9.\-])"
)
_OBFUSCATED_DOT_RE = re.compile(r"(?i)(?<=[A-Z0-9])\s*(?:\[\s*dot\s*\]|\(\s*dot\s*\)|\bdot\b)\s*(?=[A-Z0-9])")
_DISPLAY_NAME_TOKEN_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'`-]{0,29}$")
_CANDIDATE_SORT_FIELDS = {
    "id": Employee.id,
    "name": Employee.full_name,
    "email": Employee.email,
    "title": Employee.position,
    "uploaded_at": Employee.created_at,
}
_MAX_BOARD_ITEMS = 80
_MAX_BOARD_ITEM_CHARS = 220


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
    certifications = _parse_board_items(getattr(employee, "candidate_certifications", None))
    hands_on_projects = _parse_board_items(getattr(employee, "candidate_projects", None))
    return CandidateListItem(
        id=int(employee.id),
        employee_number=str(employee.employee_number or "").strip(),
        full_name=str(employee.full_name or "").strip() or "Candidate Upload",
        email=str(employee.email or "").strip(),
        phone=(str(employee.phone or "").strip() or None),
        predicted_title=(str(employee.position or "").strip() or None),
        predicted_experience_years=_experience_years_from_hire_date(getattr(employee, "hire_date", None)),
        skills=skills,
        certifications=certifications,
        hands_on_projects=hands_on_projects,
        uploaded_at=uploaded_at,
    )


def _get_candidate_or_404(db: Session, candidate_id: int) -> Employee:
    candidate = db.get(Employee, candidate_id)
    if candidate is None or str(candidate.recruitment_source or "").strip().lower() != "cv_upload":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "candidate_not_found",
                "message": "Candidate not found.",
            },
        )
    return candidate


def _normalize_skill_names(raw_skills: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in raw_skills:
        name = str(raw or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(name)
    return cleaned


def _normalize_board_items(values: list[str], *, max_items: int = _MAX_BOARD_ITEMS) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = re.sub(r"\s+", " ", str(raw or "")).strip()
        if not value:
            continue
        value = value[:_MAX_BOARD_ITEM_CHARS]
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(value)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _parse_board_items(raw: object | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return _normalize_board_items([str(item or "") for item in raw])
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
        except Exception:
            return []
        if not isinstance(decoded, list):
            return []
        return _normalize_board_items([str(item or "") for item in decoded])
    return []


def _serialize_board_items(values: list[str]) -> str | None:
    normalized = _normalize_board_items(values)
    if not normalized:
        return None
    return json.dumps(normalized, ensure_ascii=False)


def _generated_candidate_email() -> str:
    return f"cv_{uuid.uuid4().hex[:12]}@candidate.local"


def _candidate_email_alias(parsed_email: str) -> str:
    local, _, domain = str(parsed_email or "").partition("@")
    suffix = uuid.uuid4().hex[:6]
    aliased_local = f"{local}+cv{suffix}" if local else f"cv_{suffix}"
    if len(aliased_local) > 64:
        aliased_local = aliased_local[:64]
    return f"{aliased_local}@{domain or 'candidate.local'}"


def _normalize_parsed_email(value: object | None) -> str | None:
    email = str(value or "").strip().lower()
    if not email:
        return None
    if not _PARSED_EMAIL_RE.fullmatch(email):
        return None
    return email


def _extract_email_from_preview(parsed: dict) -> str | None:
    preview = str(parsed.get("preview") or "")
    if not preview:
        return None
    for variant in _contact_preview_variants(preview):
        match = _PREVIEW_EMAIL_RE.search(variant)
        if not match:
            continue
        normalized = _normalize_parsed_email(match.group(0))
        if normalized:
            return normalized
    return None


def _resolve_candidate_email(db: Session, parsed: dict) -> str:
    parsed_email = _normalize_parsed_email(parsed.get("extracted_email")) or _extract_email_from_preview(parsed)
    if parsed_email:
        duplicate = db.query(Employee.id).filter(func.lower(Employee.email) == parsed_email).first()
        if duplicate is None:
            return parsed_email
        aliased = _candidate_email_alias(parsed_email)
        duplicate_alias = db.query(Employee.id).filter(func.lower(Employee.email) == aliased.lower()).first()
        if duplicate_alias is None:
            return aliased
    return _generated_candidate_email()


def _normalize_parsed_phone(value: object | None) -> str | None:
    phone = str(value or "")
    if not phone:
        return None
    phone = phone.replace("\u200b", " ").replace("\u200c", " ").replace("\u200d", " ")
    phone = re.sub(r"\s+", " ", phone).strip(" -,:;|.")
    if not phone:
        return None
    digits = re.sub(r"\D", "", phone)
    if len(digits) < 8 or len(digits) > 15:
        return None
    if len(set(digits)) <= 1:
        return None
    return phone


def _extract_phone_from_preview(parsed: dict) -> str | None:
    preview = str(parsed.get("preview") or "")
    if not preview:
        return None
    for variant in _contact_preview_variants(preview):
        for match in _PREVIEW_PHONE_RE.finditer(variant):
            phone = _normalize_parsed_phone(match.group(0))
            if phone:
                return phone
        labeled = re.search(
            r"(?i)(?:phone|telephone|tel|mobile|gsm)\s*[:\-]?\s*([+()\d][\d\s().\-]{6,})",
            variant,
        )
        if labeled:
            phone = _normalize_parsed_phone(labeled.group(1))
            if phone:
                return phone
    return None


def _contact_preview_variants(preview: str) -> list[str]:
    variants: list[str] = []

    def _append(value: str) -> None:
        text = str(value or "").strip()
        if text and text not in variants:
            variants.append(text)

    base = str(preview or "")
    _append(base)

    normalized = _OBFUSCATED_AT_RE.sub("@", base)
    normalized = _OBFUSCATED_DOT_RE.sub(".", normalized)
    normalized = re.sub(r"\s*@\s*", "@", normalized)
    normalized = re.sub(r"\s*\.\s*", ".", normalized)
    _append(normalized)

    compact = re.sub(r"[\s\u200b\u200c\u200d]+", "", normalized)
    _append(compact)

    return variants


def _resolve_candidate_phone(parsed: dict) -> str | None:
    explicit = _normalize_parsed_phone(parsed.get("extracted_phone"))
    if explicit:
        return explicit
    return _extract_phone_from_preview(parsed)


def _extract_display_name_from_preview(parsed: dict) -> str | None:
    preview = str(parsed.get("preview") or "")
    lines = [re.sub(r"\s+", " ", line).strip() for line in preview.splitlines() if line and line.strip()]
    for line in lines[:8]:
        head = re.split(r"\s*•\s*|\s+\|\s+", line, maxsplit=1)[0].strip(" -,:;|")
        if not head:
            continue
        if "@" in head or re.search(r"\d", head):
            continue
        tokens = [tok.strip(".,;:()[]{}") for tok in head.split() if tok.strip(".,;:()[]{}")]
        if not 2 <= len(tokens) <= 5:
            continue
        if any(not _DISPLAY_NAME_TOKEN_RE.fullmatch(tok) for tok in tokens):
            continue
        capitalized = sum(1 for token in tokens if token[:1].isupper() or token.isupper())
        if capitalized >= max(2, len(tokens) - 1):
            return " ".join(tokens)
    return None


def _resolve_candidate_display_name(parsed: dict, filename: str) -> str:
    explicit = str(parsed.get("extracted_full_name") or "").strip()
    if explicit:
        return explicit
    fallback = _extract_display_name_from_preview(parsed)
    if fallback:
        return fallback
    return _display_name_from_filename(filename)


def _persist_candidate_profile(
    *,
    db: Session,
    filename: str,
    parsed: dict,
    known_skill_names: list[str],
    created_by: int | None,
) -> Employee:
    display_name = _resolve_candidate_display_name(parsed, filename)
    first_name, last_name = _split_first_last_name(display_name)
    predicted_title = str(parsed.get("predicted_title") or "").strip() or "Candidate"

    employee = Employee(
        employee_number=f"CV-{uuid.uuid4().hex[:10].upper()}",
        first_name=first_name,
        last_name=last_name,
        full_name=display_name,
        email=_resolve_candidate_email(db, parsed),
        phone=_resolve_candidate_phone(parsed),
        department="candidate pool",
        position=predicted_title,
        employment_status="candidate",
        recruitment_source="cv_upload",
        hire_date=_predicted_hire_date(_safe_float(parsed.get("predicted_experience_years"))),
        candidate_certifications=_serialize_board_items(parsed.get("certifications") or []),
        candidate_projects=_serialize_board_items(parsed.get("hands_on_projects") or []),
        created_by=created_by,
    )
    db.add(employee)
    db.flush()

    seen_skills: set[str] = set()
    extracted_rows = parsed.get("extracted_skills") or [
        {"skill": raw, "confidence": 0.6, "source": "legacy"} for raw in (parsed.get("skills") or [])
    ]
    for row in extracted_rows:
        name = str((row or {}).get("skill") if isinstance(row, dict) else row or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen_skills:
            continue
        seen_skills.add(key)

        skill = db.query(Skill).filter(func.lower(Skill.name) == key).first()
        if skill is None:
            resolved_entity = get_resolved_unknown_entity_for_term(
                db=db,
                raw_value=name,
                entity_type="skill",
            )
            if resolved_entity is not None:
                if str(resolved_entity.status or "").lower() == "approved":
                    skill = resolve_approved_skill_for_term(db=db, skill_name=name)
                if skill is None:
                    continue

        if skill is None and should_queue_skill_review(
            skill_name=name,
            source=(str(row.get("source") or "") if isinstance(row, dict) else None),
            confidence=((row or {}).get("confidence") if isinstance(row, dict) else None),
            known_skill_names=known_skill_names,
        ):
            continue
        if skill is None:
            skill = Skill(name=name, created_by=created_by)
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
                    func.lower(func.trim(func.coalesce(Employee.phone, ""))).like(f"%{term}%"),
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


@router.patch(
    "/{candidate_id}",
    response_model=CandidateListItem,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
def update_candidate(
    candidate_id: int,
    payload: CandidateUpdateRequest,
    db: Session = Depends(get_db),
):
    candidate = _get_candidate_or_404(db, candidate_id)
    updates = payload.model_dump(exclude_unset=True)

    if "full_name" in updates:
        full_name = str(updates.get("full_name") or "").strip()
        if not full_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "invalid_candidate_name", "message": "Candidate name cannot be empty."},
            )
        first_name, last_name = _split_first_last_name(full_name)
        candidate.full_name = full_name
        candidate.first_name = first_name
        candidate.last_name = last_name

    if "employee_number" in updates:
        employee_number = str(updates.get("employee_number") or "").strip()
        if not employee_number:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "invalid_candidate_id", "message": "Candidate ID cannot be empty."},
            )
        duplicate = (
            db.query(Employee)
            .filter(Employee.employee_number == employee_number, Employee.id != int(candidate.id))
            .first()
        )
        if duplicate is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "employee_number_already_exists",
                    "message": "Candidate ID already exists.",
                },
            )
        candidate.employee_number = employee_number

    if "skills" in updates:
        normalized_skills = _normalize_skill_names(updates.get("skills") or [])
        db.query(EmployeeSkill).filter(EmployeeSkill.employee_id == int(candidate.id)).delete(synchronize_session=False)
        for skill_name in normalized_skills:
            key = skill_name.lower()
            skill = db.query(Skill).filter(func.lower(Skill.name) == key).first()
            if skill is None:
                skill = Skill(name=skill_name)
                db.add(skill)
                db.flush()
            db.add(EmployeeSkill(employee_id=int(candidate.id), skill_id=int(skill.id), level=3))

    if "certifications" in updates:
        candidate.candidate_certifications = _serialize_board_items(updates.get("certifications") or [])

    if "hands_on_projects" in updates:
        candidate.candidate_projects = _serialize_board_items(updates.get("hands_on_projects") or [])

    db.commit()
    db.refresh(candidate)
    return _candidate_item_from_employee(candidate)


@router.delete(
    "/{candidate_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    responses={
        404: {"model": ErrorResponse},
    },
)
def delete_candidate(
    candidate_id: int,
    db: Session = Depends(get_db),
):
    candidate = _get_candidate_or_404(db, candidate_id)
    db.delete(candidate)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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
        candidate = None
        queued_unknowns = []
        try:
            candidate = _persist_candidate_profile(
                db=db,
                filename=(file.filename or safe_name),
                parsed=parsed,
                known_skill_names=known_skills,
                created_by=(int(getattr(current_user, "id")) if getattr(current_user, "id", None) is not None else None),
            )
        except Exception:
            db.rollback()
            parsed["degraded"] = True
            warnings = list(parsed.get("warnings") or [])
            warnings.append("Candidate profile could not be saved automatically.")
            parsed["warnings"] = warnings
            logger.exception("candidate_autosave_failure filename=%s", file.filename)

        # Initialize queuing variables
        queued_unknowns = []
        cert_project_count = 0

        if candidate is not None:
            try:
                queued_unknowns = queue_unknown_entities_for_parsed_cv(
                    db=db,
                    parsed=parsed,
                    known_skill_names=known_skills,
                    candidate_id=int(candidate.id),
                    created_by=(int(getattr(current_user, "id")) if getattr(current_user, "id", None) is not None else None),
                )
                # Also queue certifications and projects with low confidence for HITL review
                cert_project_count = queue_certs_and_projects_for_review(
                    db=db,
                    certs=parsed.get("certifications") or [],
                    projects=parsed.get("hands_on_projects") or [],
                    candidate_id=int(candidate.id),
                    created_by=(int(getattr(current_user, "id")) if getattr(current_user, "id", None) is not None else None),
                )
            except Exception:
                db.rollback()
                warnings = list(parsed.get("warnings") or [])
                warnings.append("Active learning queue could not be updated.")
                parsed["warnings"] = warnings
                logger.exception("active_learning_queue_failure filename=%s", file.filename)
                queued_unknowns = []
                cert_project_count = 0

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
            extracted_full_name=parsed.get("extracted_full_name"),
            extracted_email=parsed.get("extracted_email"),
            extracted_phone=parsed.get("extracted_phone"),
            predicted_title=parsed.get("predicted_title"),
            predicted_experience_years=parsed.get("predicted_experience_years"),
            certifications=parsed.get("certifications") or [],
            hands_on_projects=parsed.get("hands_on_projects") or [],
            project_skill_links=parsed.get("project_skill_links") or [],
            needs_review_count=len(queued_unknowns) + cert_project_count,
            queued_unknown_entities=[
                str(getattr(entity, "raw_value", "")).strip()
                for entity in queued_unknowns
                if str(getattr(entity, "raw_value", "")).strip()
            ],
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
