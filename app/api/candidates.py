import io
import logging
import time
import zipfile
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.ai.skill_canonicalization import canonicalize_skill
from app.api.auth import get_current_active_user
from app.core.config import settings
from app.core.structured_log import (
    EVENT_CV_PARSE_FAILURE,
    EVENT_CV_PARSE_METRICS,
    REASON_PARSING_FAIL,
    log_structured_event,
)
from app.db.database import get_db
from app.models.skill import Skill
from app.schemas.candidate import CandidateUploadRespose
from app.schemas.common import ErrorResponse
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


def _build_skill_id_index(skill_rows: list[tuple[int, str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for sid, name in skill_rows:
        key = canonicalize_skill(name or "")
        if key:
            out[key] = int(sid)
    return out


def _attach_skill_ids(
    extracted_rows: Any,
    skill_id_index: dict[str, int],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(extracted_rows, list):
        return out
    for row in extracted_rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        key = canonicalize_skill(str(item.get("skill", "")))
        item["skill_id"] = skill_id_index.get(key) if key else None
        out.append(item)
    return out


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

@router.post(
    "/upload_cv",
    response_model=CandidateUploadRespose,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def upload_cv(file: UploadFile = File(...), db: Session = Depends(get_db)):
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

        skill_rows = [(int(sid), str(name)) for sid, name in db.query(Skill.id, Skill.name).all() if name]
        known_skills = [name for _sid, name in skill_rows]
        skill_id_index = _build_skill_id_index(skill_rows)

        parse_started = time.perf_counter()
        parsed = parse_cv_safe(
            file_bytes=contents,
            filename=safe_name,
            known_skills=known_skills,
            min_confidence=float(settings.CV_PARSER_MIN_CONFIDENCE),
            use_semantic=bool(settings.CV_PARSER_USE_SEMANTIC),
            use_hf_ner=bool(settings.CV_PARSER_USE_HF_NER),
            use_semantic_augment=bool(settings.CV_PARSER_USE_SEMANTIC_AUGMENT),
            skill_time_budget_seconds=float(settings.CV_PARSER_SKILL_TIME_BUDGET_SECONDS),
        )
        parse_duration_ms = int((time.perf_counter() - parse_started) * 1000)
        slo_ms = int(settings.CV_PARSER_SLO_MS)
        slo_violation = parse_duration_ms > slo_ms
        if slo_violation and isinstance(parsed, dict):
            warnings = parsed.setdefault("warnings", [])
            if isinstance(warnings, list) and "parse_slo_exceeded" not in warnings:
                warnings.append("parse_slo_exceeded")
        extracted_with_ids = _attach_skill_ids(parsed.get("extracted_skills", []), skill_id_index)

        extraction_channels = parsed.get("extraction_channels", {})
        channel_counts: dict[str, int] = {}
        if isinstance(extraction_channels, dict):
            for key, value in extraction_channels.items():
                channel_counts[str(key)] = len(value) if isinstance(value, list) else 0

        log_structured_event(
            logger,
            level=logging.INFO,
            event=EVENT_CV_PARSE_METRICS,
            filename=file.filename,
            mime=sniffed,
            size_bytes=len(contents),
            parser_model_version=str(settings.CV_PARSER_MODEL_VERSION),
            duration_ms=parse_duration_ms,
            slo_ms=slo_ms,
            slo_violation=bool(slo_violation),
            degraded=bool(parsed.get("degraded", False)),
            ok=bool(parsed.get("ok", False)),
            warnings_count=len(parsed.get("warnings", []) if isinstance(parsed.get("warnings"), list) else []),
            errors_count=len(parsed.get("errors", []) if isinstance(parsed.get("errors"), list) else []),
            extracted_skill_count=len(extracted_with_ids),
            extracted_language_count=len(parsed.get("extracted_languages", []) if isinstance(parsed.get("extracted_languages"), list) else []),
            channel_counts=channel_counts,
            use_semantic=bool(settings.CV_PARSER_USE_SEMANTIC),
            use_hf_ner=bool(settings.CV_PARSER_USE_HF_NER),
            use_semantic_augment=bool(settings.CV_PARSER_USE_SEMANTIC_AUGMENT),
            skill_time_budget_seconds=float(settings.CV_PARSER_SKILL_TIME_BUDGET_SECONDS),
        )

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
            extracted_skills=extracted_with_ids,
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
