from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from app.ai.confidence_calibration import fit_platt_params
from app.ai.skill_canonicalization import canonicalize_skill
from app.services.cv_parser import detect_skills_with_confidence, extract_text


def _safe_load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_records(labels_path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
        if limit is not None and limit > 0 and len(rows) >= limit:
            break
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
        for s in row.get("expected_skills", []) or []:
            if s:
                skills.append(str(s))

    seen: set[str] = set()
    out: list[str] = []
    for s in skills:
        key = canonicalize_skill(s)
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _calibration_samples(
    records: list[dict[str, Any]],
    known_skills: list[str],
    *,
    min_confidence: float,
    use_semantic: bool,
) -> tuple[list[float], list[int], dict[str, Any]]:
    raw_scores: list[float] = []
    labels: list[int] = []

    processed = 0
    missing_files = 0
    parsed_empty = 0
    prediction_rows = 0

    for row in records:
        path = Path(str(row.get("path", ""))).expanduser()
        if not path.exists() or not path.is_file():
            missing_files += 1
            continue

        text = extract_text(path.read_bytes(), path.name)
        if not text.strip():
            parsed_empty += 1

        pred_rows = detect_skills_with_confidence(
            text=text,
            known_skills=known_skills,
            min_confidence=float(min_confidence),
            use_semantic=bool(use_semantic),
        )
        gold = {canonicalize_skill(s) for s in (row.get("expected_skills") or []) if s}
        gold.discard("")

        for pred in pred_rows:
            try:
                raw = float(pred.get("confidence", 0.0))
            except Exception:
                continue
            skill = canonicalize_skill(str(pred.get("skill", "")))
            if not skill:
                continue
            y = 1 if skill in gold else 0
            raw_scores.append(raw)
            labels.append(y)
            prediction_rows += 1

        processed += 1

    detail = {
        "records_processed": processed,
        "records_missing_file": missing_files,
        "records_empty_text": parsed_empty,
        "prediction_rows": prediction_rows,
    }
    return raw_scores, labels, detail


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Platt confidence calibration from labeled CVs")
    parser.add_argument("--labels-jsonl", required=True, help="JSONL with path + expected_skills")
    parser.add_argument("--profile", default="artifacts/data_profile.json", help="Skill profile JSON (top skills)")
    parser.add_argument("--limit", type=int, default=30, help="Max labeled CVs to use (recommended 20-30)")
    parser.add_argument("--min-confidence", type=float, default=0.50, help="Lower min confidence to keep negatives")
    parser.add_argument("--use-semantic", action="store_true", help="Enable semantic detector while collecting samples")
    parser.add_argument("--out", default="artifacts/confidence_calibration.json", help="Output calibration JSON")
    parser.add_argument(
        "--report-out",
        default="artifacts/confidence_calibration_report.json",
        help="Detailed fit report JSON",
    )
    args = parser.parse_args()

    labels_path = Path(args.labels_jsonl).expanduser()
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    records = _load_records(labels_path, int(args.limit) if args.limit and args.limit > 0 else None)
    if not records:
        raise ValueError("No labeled records found in labels-jsonl")

    profile_path = Path(args.profile).expanduser() if args.profile else None
    known_skills = _load_known_skills(profile_path, records)
    if not known_skills:
        raise ValueError("known_skills is empty - provide profile or expected_skills in labels")

    raw_scores, labels, sample_stats = _calibration_samples(
        records,
        known_skills,
        min_confidence=float(args.min_confidence),
        use_semantic=bool(args.use_semantic),
    )

    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError(
            "Calibration needs both positive and negative prediction samples. "
            f"Got positives={positives}, negatives={negatives}. "
            "Try lowering --min-confidence (e.g. 0.45-0.55) or adding more varied annotated CVs."
        )

    a, b, fit_metrics = fit_platt_params(raw_scores, labels)

    calibrated = {
        "a": round(float(a), 6),
        "b": round(float(b), 6),
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "platt_newton_1d",
        "num_samples": len(labels),
        "num_positive": positives,
        "num_negative": negatives,
    }

    report = {
        "calibration": calibrated,
        "fit_metrics": fit_metrics,
        "config": {
            "labels_jsonl": str(labels_path),
            "profile": str(profile_path) if profile_path else None,
            "records_used": len(records),
            "known_skills_count": len(known_skills),
            "min_confidence": float(args.min_confidence),
            "use_semantic": bool(args.use_semantic),
        },
        "sample_stats": sample_stats,
    }

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(calibrated, indent=2), encoding="utf-8")

    report_path = Path(args.report_out).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Saved calibration: {out_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
