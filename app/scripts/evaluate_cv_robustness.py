from __future__ import annotations

import argparse
import json
from collections import Counter
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

def _load_records(labels_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


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
        for skill in row.get("expected_skills", []) or []:
            if skill:
                skills.append(str(skill))

    seen: set[str] = set()
    out: list[str] = []
    for skill in skills:
        key = canonicalize_skill(skill)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(skill)
    return out

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CV parser robustness KPIs")
    parser.add_argument("--labels-jsonl", required=True, help="JSONL labels file with field: path")
    parser.add_argument("--profile", default="artifacts/data_profile.json", help="Skill profile JSON")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--use-semantic", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--include-per-cv", action="store_true")
    parser.add_argument("--out", default="artifacts/cv_robustness_report.json")
    return parser

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    labels_path = Path(args.labels_jsonl).expanduser()
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels JSONL not found: {labels_path}")
    records = _load_records(labels_path)
    if args.max_records is not None and args.max_records > 0:
        records = records[: int(args.max_records)]
    known_skills = _load_known_skills(Path(args.profile).expanduser() if args.profile else None, records)
    total = len(records)
    crash_count = 0
    schema_valid_count = 0
    degraded_count = 0
    empty_text_count = 0
    samples_with_errors = 0
    samples_with_warnings = 0
    warning_counter: Counter[str] = Counter()
    error_counter: Counter[str] = Counter()
    per_cv: list[dict[str, Any]] = []
    for row in records:
        raw_path = row.get("path")
        path = Path(str(raw_path)).expanduser() if raw_path else None
        sample = {
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
            crash_count += 1
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
            crash_count += 1
            sample["crashed"] = True
            sample["error"] = f"{exc.__class__.__name__}"
            if args.include_per_cv:
                per_cv.append(sample)
            continue
        try:
            CandidateUploadRespose(**payload)
            sample["schema_valid"] = True
            schema_valid_count += 1
        except Exception as exc:
            sample["schema_error"] = f"{exc.__class__.__name__}"
        warnings = [str(item) for item in (payload.get("warnings") or [])]
        errors = [str(item) for item in (payload.get("errors") or [])]

        sample["degraded"] = bool(payload.get("degraded", False))
        sample["empty_text"] = (
            bool(payload.get("text_length", 0) == 0)
            or ("empty_text" in warnings)
        )
        sample["ok"] = bool(payload.get("ok", False))
        sample["errors_count"] = len(errors)
        sample["warnings_count"] = len(warnings)
        sample["error_codes"] = errors
        sample["warning_codes"] = warnings
        if sample["degraded"]:
            degraded_count += 1
        if sample["empty_text"]:
            empty_text_count += 1
        if errors:
            samples_with_errors += 1
            error_counter.update(errors)
        if warnings:
            samples_with_warnings += 1
            warning_counter.update(warnings)

        if args.include_per_cv:
            per_cv.append(sample)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "labels_jsonl": str(labels_path),
            "num_records": total,
            "known_skills_count": len(known_skills),
            "min_confidence": float(args.min_confidence),
            "use_semantic": bool(args.use_semantic),
        },
        "totals": {
            "records": total,
            "crashes": crash_count,
            "schema_valid": schema_valid_count,
            "degraded": degraded_count,
            "empty_text": empty_text_count,
            "samples_with_errors": samples_with_errors,
            "samples_with_warnings": samples_with_warnings,
        },
        "warning_counts": dict(sorted(warning_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        "error_counts": dict(sorted(error_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        "kpis": {
            "crash_rate": _safe_float_ratio(crash_count, total),
            "schema_valid_rate": _safe_float_ratio(schema_valid_count, total),
            "degraded_rate": _safe_float_ratio(degraded_count, total),
            "empty_text_rate": _safe_float_ratio(empty_text_count, total),
        },
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
