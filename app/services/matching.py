from __future__ import annotations
import os
from datetime import date
from typing import Iterable, Optional, Sequence
from app.services.embedding_service import compute_semantic_similarity
from app.ai.preprocessing import normalize_skill_name


PERFORMANCE_MAPPING = {
    "superior": 1.0,
    "accord parfait": 0.8,
    "acceptable": 0.6,
    "inacceptable": 0.4,
    "non applicable": 0.2,
}
_TRUTHY_VALUES = {"1", "true", "yes", "on"}
_embedding_service = None

def performance_weight(score: Optional[str]) -> float:
    normalized = (score or "").strip().lower()
    return PERFORMANCE_MAPPING.get(normalized, 0.0)


def _extract_employee_skills(employee) -> list[str]:
    names: list[str] = []
    for item in getattr(employee, "skills", []) or []:
        if hasattr(item, "name") and item.name:
            names.append(str(item.name))
            continue

        linked_skill = getattr(item, "skill", None)
        linked_name = getattr(linked_skill, "name", None)
        if linked_name:
            names.append(str(linked_name))

    return names


def _experience_years(employee) -> int:
    if not getattr(employee, "hire_date", None):
        return 0

    return max(0, date.today().year - employee.hire_date.year)


def _experience_score(employee, min_experience: int = 0) -> float:
    years = _experience_years(employee)
    if min_experience > 0:
        return min(years / float(min_experience), 1.0)
    return min(years / 10.0, 1.0)


def _skill_overlap(required_skills: Sequence[str], employee_skills: Sequence[str]) -> float:
    required_set = _normalize_skill_set(required_skills)
    if not required_set:
        return 0.0
    employee_set = _normalize_skill_set(employee_skills)
    overlap = len(required_set & employee_set)
    return overlap / len(required_set)


def _employee_profile_text(employee, employee_skills: Iterable[str]) -> str:
    parts = [
        str(getattr(employee, "position", "") or ""),
        str(getattr(employee, "department", "") or ""),
        " ".join(employee_skills),
    ]
    return " ".join(part for part in parts if part).strip()


def _is_ai_matching_enabled() -> bool:
    return os.getenv("ENABLE_AI_MATCHING", "false").strip().lower() in _TRUTHY_VALUES


def _get_embedding_service():
    global _embedding_service
    if _embedding_service is None:
        from app.services.embedding_service import EmbeddingService
        _embedding_service = EmbeddingService()
    return _embedding_service


def _semantic_similarity(target_text: str, employee_text: str) -> float:
    if not _is_ai_matching_enabled():
        return 0.0

    if not target_text or not employee_text:
        return 0.0

    try:
        service = _get_embedding_service()
        vec1 = service.generate_embedding(target_text)
        vec2 = service.generate_embedding(employee_text)
        return compute_semantic_similarity(vec1, vec2)
    except Exception:
        # Keep ranking functional even when embedding model fails.
        return 0.0


def calculate_weighted_score(
    employee,
    job_title: str,
    required_skills: Optional[Sequence[str]] = None,
    min_experience: int = 0,
) -> dict:
    required = list(required_skills or [])
    employee_skills = _extract_employee_skills(employee)

    target_text = " ".join([job_title, *required]).strip()
    employee_text = _employee_profile_text(employee, employee_skills)

    semantic_similarity = _semantic_similarity(target_text, employee_text)
    skill_overlap = _skill_overlap(required, employee_skills)
    experience_score = _experience_score(employee, min_experience)
    performance_score = performance_weight(getattr(employee, "performance_score", None))

    if _is_ai_matching_enabled():
        final_score = (
            0.4 * semantic_similarity
            + 0.3 * skill_overlap
            + 0.2 * experience_score
            + 0.1 * performance_score
        )
    else:
        # Non-AI baseline for stable backend testing. since i disabled the AI for improvement
        final_score = (
            0.5 * skill_overlap
            + 0.35 * experience_score
            + 0.15 * performance_score
        )

    return {
        "semantic_similarity": round(semantic_similarity, 4),
        "skill_overlap": round(skill_overlap, 4),
        "experience_score": round(experience_score, 4),
        "performance_score": round(performance_score, 4),
        "final_score": round(final_score, 4),
        "total": round(final_score * 100, 2),
    }


def calculate_match_score(
    employee,
    target_job: str,
    required_skills: Optional[Sequence[str]] = None,
    min_experience: int = 0,
):
    result = calculate_weighted_score(
        employee=employee,
        job_title=target_job,
        required_skills=required_skills,
        min_experience=min_experience,
    )

    # Keep backward compatibility with endpoints expecting a numeric score.
    if required_skills is not None:
        return result["total"]

    return result

def _normalize_skill_set(skills: Sequence[str]) -> set[str]:
    out: set[str] = set()
    for raw in skills or []:
        normalized = normalize_skill_name(raw)
        if normalized:
            out.add(normalized)
    return out
