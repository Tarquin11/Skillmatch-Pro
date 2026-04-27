from __future__ import annotations

import json
import re
from typing import Any, Iterable

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.ai.skill_canonicalization import canonicalize_skill
from app.core.config import settings
from app.models.active_learning import EntityReview, UnknownEntity
from app.models.skill import Skill


_REVIEW_SOURCE_PREFIXES = (
    "cv_section:",
    "ner_span:",
    "sentence_",
    "semantic_augment",
)
_CONTEXT_CHARS = 280
_MAX_RAW_VALUE_CHARS = 500


def normalize_learning_value(raw: str | None, *, entity_type: str = "skill") -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    if entity_type == "skill":
        normalized = canonicalize_skill(value)
    else:
        normalized = value.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def _safe_confidence(value: Any) -> float | None:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, conf))


def _clean_evidence(values: Iterable[Any]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = re.sub(r"\s+", " ", str(raw or "")).strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(value[:_CONTEXT_CHARS])
        if len(cleaned) >= 3:
            break
    return cleaned


def should_queue_skill_review(
    *,
    skill_name: str,
    source: str | None,
    confidence: float | None,
    known_skill_names: Iterable[str],
) -> bool:
    normalized = normalize_learning_value(skill_name, entity_type="skill")
    if not normalized:
        return False

    known = {normalize_learning_value(name, entity_type="skill") for name in known_skill_names if str(name or "").strip()}
    if normalized in known:
        return False

    conf = _safe_confidence(confidence)
    if conf is not None and conf < float(settings.ACTIVE_LEARNING_REVIEW_THRESHOLD):
        return True

    source_key = str(source or "").strip().lower()
    return any(source_key.startswith(prefix) for prefix in _REVIEW_SOURCE_PREFIXES)


def queue_unknown_entities_for_parsed_cv(
    *,
    db: Session,
    parsed: dict[str, Any],
    known_skill_names: Iterable[str],
    candidate_id: int | None,
    created_by: int | None,
) -> list[UnknownEntity]:
    if not bool(settings.ACTIVE_LEARNING_ENABLED):
        return []

    queued: list[UnknownEntity] = []
    seen_terms: set[str] = set()
    preview = re.sub(r"\s+", " ", str(parsed.get("preview") or "")).strip()
    rows = parsed.get("extracted_skills") or []

    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_value = str(row.get("skill") or "").strip()
        source = str(row.get("source") or "").strip()
        confidence = _safe_confidence(row.get("confidence"))
        if not should_queue_skill_review(
            skill_name=raw_value,
            source=source,
            confidence=confidence,
            known_skill_names=known_skill_names,
        ):
            continue

        normalized = normalize_learning_value(raw_value, entity_type="skill")
        if not normalized or normalized in seen_terms:
            continue
        seen_terms.add(normalized)

        evidence = _clean_evidence(row.get("evidence") or [])
        context_excerpt = evidence[0] if evidence else preview[:_CONTEXT_CHARS] or None
        entity = upsert_unknown_entity(
            db=db,
            raw_value=raw_value,
            normalized_value=normalized,
            entity_type_guess="skill",
            source=source or None,
            confidence=confidence,
            confidence_band=(str(row.get("confidence_band") or "").strip() or None),
            evidence=evidence,
            context_excerpt=context_excerpt,
            candidate_id=candidate_id,
            created_by=created_by,
        )
        queued.append(entity)

    if queued:
        db.commit()
        for entity in queued:
            db.refresh(entity)
    return queued


def queue_certs_and_projects_for_review(
    *,
    db: Session,
    certs: list[str] | None,
    projects: list[str] | None,
    candidate_id: int | None,
    created_by: int | None,
) -> int:
    """Queue certifications and projects with low confidence for HITL review."""
    if not bool(settings.ACTIVE_LEARNING_ENABLED):
        return 0

    queued_count = 0
    seen_terms: set[str] = set()

    # Low confidence (0.60) for certifications since they are often short/ambiguous and more likely to be noisy, but still worth reviewing in HITL --- IGNORE ---
    for cert_value in (certs or []):
        cert_value = str(cert_value or "").strip()
        if not cert_value:
            continue

        normalized = normalize_learning_value(cert_value, entity_type="certification")
        if not normalized or normalized in seen_terms:
            continue
        seen_terms.add(normalized)

        entity = upsert_unknown_entity(
            db=db,
            raw_value=cert_value,
            normalized_value=normalized,
            entity_type_guess="certification",
            source="cert_section",
            confidence=0.60,
            confidence_band="low",
            evidence=[cert_value],
            context_excerpt=cert_value[:_CONTEXT_CHARS],
            candidate_id=candidate_id,
            created_by=created_by,
        )
        queued_count += 1

    # Queue projects - assign low confidence (0.65) so all go to HITL review
    for project_value in (projects or []):
        project_value = str(project_value or "").strip()
        if not project_value:
            continue

        normalized = normalize_learning_value(project_value, entity_type="project")
        if not normalized or normalized in seen_terms:
            continue
        seen_terms.add(normalized)

        entity = upsert_unknown_entity(
            db=db,
            raw_value=project_value,
            normalized_value=normalized,
            entity_type_guess="project",
            source="project_section",
            confidence=0.65,
            confidence_band="low",
            evidence=[project_value],
            context_excerpt=project_value[:_CONTEXT_CHARS],
            candidate_id=candidate_id,
            created_by=created_by,
        )
        queued_count += 1

    if queued_count > 0:
        db.commit()

    return queued_count


def upsert_unknown_entity(
    *,
    db: Session,
    raw_value: str,
    normalized_value: str,
    entity_type_guess: str,
    source: str | None,
    confidence: float | None,
    confidence_band: str | None,
    evidence: list[str],
    context_excerpt: str | None,
    candidate_id: int | None,
    created_by: int | None,
) -> UnknownEntity:
    entity = db.query(UnknownEntity).filter(UnknownEntity.normalized_value == normalized_value).first()
    if entity is None:
        entity = UnknownEntity(
            raw_value=str(raw_value or "").strip()[:_MAX_RAW_VALUE_CHARS],
            normalized_value=normalized_value,
            entity_type_guess=entity_type_guess,
            status="pending",
            source=source,
            confidence=confidence,
            confidence_band=confidence_band,
            evidence_json=json.dumps(evidence, ensure_ascii=False),
            context_excerpt=(str(context_excerpt or "").strip()[:_CONTEXT_CHARS] or None),
            occurrence_count=1,
            candidate_id=candidate_id,
            created_by=created_by,
        )
        db.add(entity)
        db.flush()
        return entity

    entity.occurrence_count = int(entity.occurrence_count or 0) + 1
    if str(entity.entity_type_guess or "unknown") == "unknown" and entity_type_guess != "unknown":
        entity.entity_type_guess = entity_type_guess
    if candidate_id is not None and entity.candidate_id is None:
        entity.candidate_id = candidate_id
    if source and not entity.source:
        entity.source = source
    if confidence is not None and (entity.confidence is None or float(confidence) > float(entity.confidence)):
        entity.confidence = confidence
        entity.confidence_band = confidence_band or entity.confidence_band
    if evidence:
        entity.evidence_json = json.dumps(evidence, ensure_ascii=False)
    if context_excerpt:
        entity.context_excerpt = str(context_excerpt).strip()[:_CONTEXT_CHARS]
    db.flush()
    return entity


def serialize_unknown_entity(entity: UnknownEntity, *, include_reviews: bool = True) -> dict[str, Any]:
    try:
        evidence = json.loads(entity.evidence_json or "[]")
    except Exception:
        evidence = []
    if not isinstance(evidence, list):
        evidence = []
    reviews = []
    if include_reviews:
        for review in entity.reviews or []:
            reviews.append(
                {
                    "id": int(review.id),
                    "unknown_entity_id": int(review.unknown_entity_id),
                    "reviewer_id": int(review.reviewer_id),
                    "decision": str(review.decision),
                    "entity_type": str(review.entity_type),
                    "canonical_value": (str(review.canonical_value) if review.canonical_value is not None else None),
                    "notes": (str(review.notes) if review.notes is not None else None),
                    "promoted_skill_id": (int(review.promoted_skill_id) if review.promoted_skill_id is not None else None),
                    "created_at": review.created_at,
                }
            )
    return {
        "id": int(entity.id),
        "raw_value": str(entity.raw_value),
        "normalized_value": str(entity.normalized_value),
        "entity_type_guess": str(entity.entity_type_guess),
        "resolved_entity_type": (str(entity.resolved_entity_type) if entity.resolved_entity_type else None),
        "status": str(entity.status),
        "source": (str(entity.source) if entity.source else None),
        "confidence": _safe_confidence(entity.confidence),
        "confidence_band": (str(entity.confidence_band) if entity.confidence_band else None),
        "evidence": _clean_evidence(evidence),
        "context_excerpt": (str(entity.context_excerpt) if entity.context_excerpt else None),
        "occurrence_count": max(1, int(entity.occurrence_count or 1)),
        "candidate_id": (int(entity.candidate_id) if entity.candidate_id is not None else None),
        "canonical_skill_id": (int(entity.canonical_skill_id) if entity.canonical_skill_id is not None else None),
        "created_at": entity.created_at,
        "updated_at": entity.updated_at,
        "reviews": reviews,
    }


def review_unknown_entity(
    *,
    db: Session,
    entity: UnknownEntity,
    reviewer_id: int,
    decision: str,
    entity_type: str,
    canonical_value: str | None,
    notes: str | None,
) -> UnknownEntity:
    promoted_skill: Skill | None = None
    canonical = str(canonical_value or entity.raw_value or "").strip() or None

    if decision == "approved" and entity_type == "skill":
        skill_name = canonical or str(entity.raw_value or "").strip()
        existing_skill = (
            db.query(Skill)
            .filter(func.lower(Skill.name) == skill_name.lower())
            .first()
        )
        if existing_skill is None:
            existing_skill = Skill(name=skill_name, created_by=reviewer_id)
            db.add(existing_skill)
            db.flush()
        promoted_skill = existing_skill
        entity.canonical_skill_id = int(existing_skill.id)
    elif decision == "rejected":
        entity.canonical_skill_id = None

    entity.status = decision
    entity.resolved_entity_type = entity_type

    review = EntityReview(
        unknown_entity_id=int(entity.id),
        reviewer_id=reviewer_id,
        decision=decision,
        entity_type=entity_type,
        canonical_value=canonical,
        notes=(str(notes).strip() or None) if notes is not None else None,
        promoted_skill_id=(int(promoted_skill.id) if promoted_skill is not None else None),
        created_by=reviewer_id,
    )
    db.add(review)
    db.commit()
    db.refresh(entity)
    return entity
