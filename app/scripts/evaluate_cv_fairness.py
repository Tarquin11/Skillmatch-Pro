from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.ai.skill_canonicalization import canonicalize_skill
from app.schemas.candidate import CandidateUploadRespose
from app.services.cv_parser import parse_cv_safe


def _safe_float_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _safe_load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_records(labels_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for labels_path in labels_paths:
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                continue
            payload["__labels_path__"] = str(labels_path)
            rows.append(payload)
    return rows


def _resolve_cv_path(row: dict[str, Any], labels_path: Path) -> Path | None:
    raw_path = row.get("path")
    if raw_path:
        return Path(str(raw_path)).expanduser()

    raw_file = row.get("file")
    if not raw_file:
        return None

    p = Path(str(raw_file)).expanduser()
    if not p.is_absolute():
        p = labels_path.parent / p
    return p


def _extract_group_value(row: dict[str, Any], group_field: str) -> str:
    current: Any = row
    for part in [item.strip() for item in group_field.split(".") if item.strip()]:
        if not isinstance(current, dict):
            current = None
            break
        current = current.get(part)

    if current is None:
        return "unknown"
    value = str(current).strip()
    return value if value else "unknown"


def _load_known_skills(profile_path: Path | None, records: list[dict[str, Any]]) -> list[str]:
    skills: list[str] = []
    if profile_path and profile_path.exists():
        data = _safe_load_json(profile_path)
        for name, _count in data.get("top_employee_skills", []):
            if name:
                skills.append(str(name))
        for name, _count in data.get("top_job_skills", []):
            if name:
                skills.append(str(name))

    for row in records:
        for item in row.get("expected_skills", []) or []:
            if item:
                skills.append(str(item))
        labels = row.get("labels")
        if isinstance(labels, dict):
            for item in labels.get("skills", []) or []:
                if item:
                    skills.append(str(item))
            for item in labels.get("tools", []) or []:
                if item:
                    skills.append(str(item))

    seen: set[str] = set()
    out: list[str] = []
    for skill in skills:
        key = canonicalize_skill(skill)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(skill)
    return out


def _group_template() -> dict[str, Any]:
    return {
        "records": 0,
        "crashes": 0,
        "schema_valid": 0,
        "degraded": 0,
        "empty_text": 0,
        "ok": 0,
        "samples_with_errors": 0,
        "samples_with_warnings": 0,
        "warning_counts": Counter(),
        "error_counts": Counter(),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CV parser fairness by cohort groups")
    parser.add_argument(
        "--labels-jsonl",
        action="append",
        required=True,
        help="JSONL labels file (repeatable). Supports records with path=... or file=... fields.",
    )
    parser.add_argument("--group-field", default="identity.industry", help="Dot-path field used for cohort grouping.")
    parser.add_argument("--profile", default="artifacts/data_profile.json", help="Skill profile JSON")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--use-semantic", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--min-group-size", type=int, default=3)
    parser.add_argument("--include-per-cv", action="store_true")
    parser.add_argument("--out", default="artifacts/cv_fairness_report.json")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    labels_paths = [Path(item).expanduser() for item in (args.labels_jsonl or [])]
    for labels_path in labels_paths:
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels JSONL not found: {labels_path}")

    records = _load_records(labels_paths)
    if args.max_records is not None and args.max_records > 0:
        records = records[: int(args.max_records)]

    known_skills = _load_known_skills(Path(args.profile).expanduser() if args.profile else None, records)

    by_group: dict[str, dict[str, Any]] = defaultdict(_group_template)
    per_cv: list[dict[str, Any]] = []

    for row in records:
        labels_path = Path(str(row.get("__labels_path__", ""))).expanduser()
        path = _resolve_cv_path(row, labels_path)
        group = _extract_group_value(row, str(args.group_field))
        stats = by_group[group]
        stats["records"] += 1

        sample = {
            "group": group,
            "path": str(path) if path else "",
            "crashed": False,
            "schema_valid": False,
            "degraded": False,
            "empty_text": False,
            "ok": False,
            "errors_count": 0,
            "warnings_count": 0,
        }

        if path is None or not path.exists():
            stats["crashes"] += 1
            sample["crashed"] = True
            sample["error"] = "file_not_found"
            if args.include_per_cv:
                per_cv.append(sample)
            continue

        try:
            payload = parse_cv_safe(
                file_bytes=path.read_bytes(),
                filename=path.name,
                known_skills=known_skills,
                min_confidence=float(args.min_confidence),
                use_semantic=bool(args.use_semantic),
            )
        except Exception as exc:
            stats["crashes"] += 1
            sample["crashed"] = True
            sample["error"] = f"{exc.__class__.__name__}"
            if args.include_per_cv:
                per_cv.append(sample)
            continue

        try:
            CandidateUploadRespose(**payload)
            sample["schema_valid"] = True
            stats["schema_valid"] += 1
        except Exception as exc:
            sample["schema_error"] = f"{exc.__class__.__name__}"

        warnings = [str(item) for item in (payload.get("warnings") or [])]
        errors = [str(item) for item in (payload.get("errors") or [])]
        sample["degraded"] = bool(payload.get("degraded", False))
        sample["empty_text"] = bool(payload.get("text_length", 0) == 0) or ("empty_text" in warnings)
        sample["ok"] = bool(payload.get("ok", False))
        sample["errors_count"] = len(errors)
        sample["warnings_count"] = len(warnings)

        if sample["degraded"]:
            stats["degraded"] += 1
        if sample["empty_text"]:
            stats["empty_text"] += 1
        if sample["ok"]:
            stats["ok"] += 1
        if errors:
            stats["samples_with_errors"] += 1
            stats["error_counts"].update(errors)
        if warnings:
            stats["samples_with_warnings"] += 1
            stats["warning_counts"].update(warnings)

        if args.include_per_cv:
            per_cv.append(sample)

    group_summary: dict[str, Any] = {}
    for group, stats in sorted(by_group.items(), key=lambda kv: kv[0]):
        n = int(stats["records"])
        group_summary[group] = {
            "records": n,
            "totals": {
                "crashes": int(stats["crashes"]),
                "schema_valid": int(stats["schema_valid"]),
                "degraded": int(stats["degraded"]),
                "empty_text": int(stats["empty_text"]),
                "ok": int(stats["ok"]),
                "samples_with_errors": int(stats["samples_with_errors"]),
                "samples_with_warnings": int(stats["samples_with_warnings"]),
            },
            "warning_counts": dict(
                sorted(
                    stats["warning_counts"].items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )
            ),
            "error_counts": dict(
                sorted(
                    stats["error_counts"].items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )
            ),
            "kpis": {
                "crash_rate": _safe_float_ratio(int(stats["crashes"]), n),
                "schema_valid_rate": _safe_float_ratio(int(stats["schema_valid"]), n),
                "degraded_rate": _safe_float_ratio(int(stats["degraded"]), n),
                "empty_text_rate": _safe_float_ratio(int(stats["empty_text"]), n),
                "ok_rate": _safe_float_ratio(int(stats["ok"]), n),
            },
        }

    min_group_size = max(1, int(args.min_group_size))
    eligible_groups = [g for g, s in group_summary.items() if int(s["records"]) >= min_group_size]

    metric_names = (
        "crash_rate",
        "schema_valid_rate",
        "degraded_rate",
        "empty_text_rate",
        "ok_rate",
    )
    disparities: dict[str, float | None] = {}
    for metric_name in metric_names:
        values = [float(group_summary[g]["kpis"][metric_name]) for g in eligible_groups]
        if len(values) < 2:
            disparities[f"{metric_name}_gap"] = None
            continue
        disparities[f"{metric_name}_gap"] = round(max(values) - min(values), 6)

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "labels_jsonl": [str(p) for p in labels_paths],
            "group_field": str(args.group_field),
            "num_records": len(records),
            "known_skills_count": len(known_skills),
            "min_confidence": float(args.min_confidence),
            "use_semantic": bool(args.use_semantic),
            "min_group_size": min_group_size,
        },
        "totals": {
            "records": len(records),
            "groups_detected": len(group_summary),
            "eligible_group_count": len(eligible_groups),
            "eligible_groups": eligible_groups,
        },
        "groups": group_summary,
        "disparities": disparities,
    }

    if args.include_per_cv:
        report["per_cv"] = per_cv

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report to: {out_path}")


if __name__ == "__main__":
    main()
