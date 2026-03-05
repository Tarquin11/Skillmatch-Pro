from __future__ import annotations
import re
from datetime import date, datetime
from typing import Any,Iterable

SKILL_SPLIT_RE = re.compile(r"[,\|;/]+")
NON_ALNUM_SKILL_RE= re.compile(r"[^a-z0-9+.#/\-\s]")
VERSION_SUFFIX_RE=re.compile(r"(?<=\D)\d+(\.\d+)*$")

PERFORMANCE_MAP = {
    "superior": 1.0,
    "accord parfait": 0.8,
    "acceptable": 0.6,
    "inacceptable": 0.4,
    "non applicable": 0.2,
}

def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()

def parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()

    raw = to_text(value)
    if not raw:
        return None

    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def years_between(start: date | None, end: date | None = None) -> float:
    if start is None:
        return 0.0
    end = end or date.today()
    days = max(0, (end - start).days)
    return round(days / 365.25, 2)


def normalize_skill_name(skill: Any) -> str:
    value = to_text(skill).lower().replace("&"," and ")
    value = NON_ALNUM_SKILL_RE.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = VERSION_SUFFIX_RE.sub("", value).strip()
    return value


def parse_skills(raw: Any) -> set[str]:
    if raw is None:
        return set()

    if isinstance(raw, str):
        parts = SKILL_SPLIT_RE.split(raw)
        return {normalize_skill_name(p) for p in parts if normalize_skill_name(p)}

    if isinstance(raw, (list, tuple, set)):
        out: set[str] = set()
        for item in raw:
            if isinstance(item, dict):
                name = item.get("name") or item.get("skill") or ""
                if isinstance(name, dict):
                    name = name.get("name", "")
                n = normalize_skill_name(name)
                if n:
                    out.add(n)
                continue

            # SQLAlchemy relationship object support
            skill_obj = getattr(item, "skill", None)
            skill_name = getattr(skill_obj, "name", None) or getattr(item, "name", None) or item
            n = normalize_skill_name(skill_name)
            if n:
                out.add(n)
        return out

    return set()


def normalize_performance(raw: Any) -> float:
    text = to_text(raw).lower()
    if not text:
        return 0.0
    if text in PERFORMANCE_MAP:
        return PERFORMANCE_MAP[text]
    try:
        val = float(text)
        if val > 1.0:
            val = val / 100.0 if val <= 100 else 1.0
        return max(0.0, min(1.0, val))
    except (TypeError, ValueError):
        return 0.0


def normalize_scalar(raw: Any, max_scale: float = 5.0) -> float:
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if max_scale <= 0:
        return 0.0
    if val > 1.0:
        val = val / max_scale
    return max(0.0, min(1.0, val))


def preprocess_employee(employee: Any) -> dict[str, Any]:
    hire_date = parse_date(_get(employee, "hire_date"))
    termination_date = parse_date(_get(employee, "termination_date"))

    employment_status = to_text(_get(employee, "employment_status")).lower()
    currently_active = 1.0 if (not termination_date and employment_status != "terminated") else 0.0

    skills = parse_skills(_get(employee, "skills", []))

    return {
        "id": _get(employee, "id"),
        "full_name": to_text(_get(employee, "full_name"))
        or f"{to_text(_get(employee, 'first_name'))} {to_text(_get(employee, 'last_name'))}".strip(),
        "department": to_text(_get(employee, "departement") or _get(employee, "department")).lower(),
        "position": to_text(_get(employee, "position")).lower(),
        "skills": skills,
        "hire_date": hire_date,
        "termination_date": termination_date,
        "experience_years": years_between(hire_date),
        "tenure_years": years_between(hire_date, termination_date or date.today()),
        "currently_active": currently_active,
        "performance_score": normalize_performance(_get(employee, "performance_score")),
        "engagement_score": normalize_scalar(_get(employee, "engagement_survey"), max_scale=5.0),
        "satisfaction_score": normalize_scalar(_get(employee, "emp_satisfaction"), max_scale=5.0),
    }


def preprocess_job(job: Any) -> dict[str, Any]:
    req_skills = _get(job, "required_skills", [])
    # For SQLAlchemy JobSkill rows
    skill_names: list[str] = []
    for item in req_skills or []:
        name = None
        if isinstance(item, dict):
            name = item.get("name")
            if not name and isinstance(item.get("skill"), dict):
                name = item["skill"].get("name")
        else:
            linked_skill = getattr(item, "skill", None)
            name = getattr(linked_skill, "name", None) or getattr(item, "name", None)
        if name:
            skill_names.append(str(name))

    # Allow API payload fallback
    payload_required = _get(job, "required_skills", [])
    if isinstance(payload_required, list) and payload_required and not skill_names:
        skill_names = [str(s) for s in payload_required]

    min_exp = _get(job, "min_experience")
    if min_exp is None:
        min_exp = _get(job, "required_experience_years", 0)

    try:
        min_exp_value = float(min_exp or 0)
    except (TypeError, ValueError):
        min_exp_value = 0.0

    return {
        "id": _get(job, "id"),
        "title": to_text(_get(job, "title")).lower(),
        "description": to_text(_get(job, "description")).lower(),
        "department": to_text(_get(job, "departement") or _get(job, "department")).lower(),
        "required_skills": parse_skills(skill_names),
        "required_experience_years": max(0.0, min_exp_value),
    }


def preprocess_training_pairs(pairs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in pairs:
        employee = preprocess_employee(row["employee"])
        job = preprocess_job(row["job"])
        label = int(row["label"])
        out.append({"employee": employee, "job": job, "label": label})
    return out