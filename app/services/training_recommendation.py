from __future__ import annotations
from typing import Any, Sequence
from app.ai.preprocessing import normalize_skill_name, parse_skills

def _ordered_unique_normalized(skills: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in skills or []:
        normalized = normalize_skill_name(raw)
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def _display_name(skill: str) -> str:
    if "/" in skill or len(skill) <= 3:
        return skill.upper()
    return skill.title()


def _course_titles(skill: str, job_title: str) -> list[str]:
    label = _display_name(skill)
    context = f" for {job_title.strip()}" if (job_title or "").strip() else ""
    return [
        f"{label} Fundamentals{context}",
        f"Applied {label} Projects{context}",
        f"{label} Practice Lab{context}",
    ]


def build_training_recommendations(
    *,
    job_title: str,
    required_skills: Sequence[str],
    owned_skills: Sequence[str],
    top_k: int = 3,
) -> dict[str, Any]:
    ordered_required = _ordered_unique_normalized(required_skills)
    required_set = set(ordered_required)
    owned = parse_skills(list(owned_skills or []))

    missing = [skill for skill in ordered_required if skill not in owned]
    matched = [skill for skill in ordered_required if skill in owned]

    if not required_set:
        return {
            "required_skills_normalized": [],
            "matched_skills": [],
            "missing_skills": [],
            "skill_gap_ratio": 0.0,
            "learning_recommendations": [],
        }

    scoped_missing = missing[: max(top_k, 0)]
    denominator = max(len(scoped_missing) - 1, 1)

    learning_recommendations: list[dict[str, Any]] = []
    for idx, skill in enumerate(scoped_missing):
        priority = round(1.0 - ((idx / denominator) * 0.4), 2) if scoped_missing else 0.0
        learning_recommendations.append(
            {
                "missing_skill": skill,
                "learning_topic": f"{_display_name(skill)} for {job_title}".strip(),
                "recommended_courses": _course_titles(skill, job_title)[:3],
                "priority_score": priority,
            }
        )
    return {
        "required_skills_normalized": ordered_required,
        "matched_skills": matched,
        "missing_skills": missing,
        "skill_gap_ratio": round(len(missing) / len(required_set), 4),
        "learning_recommendations": learning_recommendations,
    }
