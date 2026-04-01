import argparse
import json
import re
from pathlib import Path
from typing import Any

from app.ai.skill_canonicalization import canonicalize_skill
from app.services.cv_parser import (
    detect_experience_years,
    detect_skills_with_confidence,
    detect_title,
    extract_text,
)


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _norm_skill(value: str) -> str:
    return canonicalize_skill(value or "")


def _safe_load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
        key = _norm_skill(s)
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _eval_one(
    record: dict[str, Any],
    known_skills: list[str],
    min_confidence: float,
    use_semantic: bool,
    experience_tolerance: float,
) -> dict[str, Any]:
    path = Path(str(record["path"])).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"CV file not found: {path}")

    text = extract_text(path.read_bytes(), path.name)
    rows = detect_skills_with_confidence(
        text=text,
        known_skills=known_skills,
        min_confidence=min_confidence,
        use_semantic=use_semantic,
    )

    pred_skills = {_norm_skill(r["skill"]) for r in rows if r.get("skill")}
    gold_skills = {_norm_skill(s) for s in (record.get("expected_skills") or []) if s}
    pred_skills.discard("")
    gold_skills.discard("")

    tp = sorted(pred_skills & gold_skills)
    fp = sorted(pred_skills - gold_skills)
    fn = sorted(gold_skills - pred_skills)

    expected_title = record.get("expected_title")
    predicted_title = detect_title(text)
    title_ok = None
    if expected_title is not None:
        a = _norm_text(str(expected_title))
        b = _norm_text(str(predicted_title or ""))
        title_ok = bool(a and b and (a in b or b in a))

    expected_exp = record.get("expected_experience_years")
    predicted_exp = detect_experience_years(text)
    exp_ok = None
    exp_abs_err = None
    if expected_exp is not None:
        expected_exp = float(expected_exp)
        if predicted_exp is not None:
            exp_abs_err = abs(predicted_exp - expected_exp)
            exp_ok = exp_abs_err <= experience_tolerance
        else:
            exp_ok = False

    return {
        "path": str(path),
        "expected_skills_count": len(gold_skills),
        "predicted_skills_count": len(pred_skills),
        "skills": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "metrics": _prf(len(tp), len(fp), len(fn)),
        },
        "title": {
            "expected": expected_title,
            "predicted": predicted_title,
            "ok": title_ok,
        },
        "experience_years": {
            "expected": expected_exp,
            "predicted": predicted_exp,
            "abs_error": exp_abs_err,
            "ok_within_tolerance": exp_ok,
            "tolerance": experience_tolerance,
        },
        "preview": (text or "")[:250],
    }


def _load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.labels_jsonl:
        rows: list[dict[str, Any]] = []
        for line in Path(args.labels_jsonl).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    if args.cv:
        return [
            {
                "path": args.cv,
                "expected_skills": [s.strip() for s in (args.expected_skills or "").split(",") if s.strip()],
                "expected_title": args.expected_title,
                "expected_experience_years": args.expected_experience_years,
            }
        ]
    raise ValueError("Provide either --labels-jsonl or --cv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CV extraction quality on labeled CVs")
    parser.add_argument("--labels-jsonl", help="JSONL file: path, expected_skills, expected_title, expected_experience_years")
    parser.add_argument("--cv", help="Single CV file path")
    parser.add_argument("--expected-skills", default="", help="Comma-separated expected skills (single CV mode)")
    parser.add_argument("--expected-title", default=None, help="Expected title (single CV mode)")
    parser.add_argument("--expected-experience-years", type=float, default=None, help="Expected years of experience (single CV mode)")
    parser.add_argument("--profile", default="artifacts/data_profile.json", help="Skill profile JSON (top skills)")
    parser.add_argument("--min-confidence", type=float, default=0.95)
    parser.add_argument("--use-semantic", action="store_true")
    parser.add_argument("--experience-tolerance", type=float, default=1.0)
    parser.add_argument("--out", default="artifacts/cv_extraction_report.json")
    args = parser.parse_args()

    records = _load_records(args)
    profile_path = Path(args.profile) if args.profile else None
    known_skills = _load_known_skills(profile_path, records)

    per_cv: list[dict[str, Any]] = []
    agg_tp = agg_fp = agg_fn = 0
    title_total = title_ok = 0
    exp_total = exp_ok = 0
    exp_errors: list[float] = []

    for rec in records:
        one = _eval_one(
            record=rec,
            known_skills=known_skills,
            min_confidence=float(args.min_confidence),
            use_semantic=bool(args.use_semantic),
            experience_tolerance=float(args.experience_tolerance),
        )
        per_cv.append(one)
        agg_tp += len(one["skills"]["tp"])
        agg_fp += len(one["skills"]["fp"])
        agg_fn += len(one["skills"]["fn"])

        if one["title"]["ok"] is not None:
            title_total += 1
            title_ok += int(bool(one["title"]["ok"]))

        if one["experience_years"]["ok_within_tolerance"] is not None:
            exp_total += 1
            exp_ok += int(bool(one["experience_years"]["ok_within_tolerance"]))
        if one["experience_years"]["abs_error"] is not None:
            exp_errors.append(float(one["experience_years"]["abs_error"]))

    report = {
        "config": {
            "num_records": len(records),
            "known_skills_count": len(known_skills),
            "min_confidence": float(args.min_confidence),
            "use_semantic": bool(args.use_semantic),
            "experience_tolerance": float(args.experience_tolerance),
        },
        "aggregate": {
            "skills_micro": _prf(agg_tp, agg_fp, agg_fn),
            "skills_counts": {"tp": agg_tp, "fp": agg_fp, "fn": agg_fn},
            "title_accuracy": (title_ok / title_total) if title_total else None,
            "experience_within_tolerance_rate": (exp_ok / exp_total) if exp_total else None,
            "experience_mae": (sum(exp_errors) / len(exp_errors)) if exp_errors else None,
        },
        "per_cv": per_cv,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report to: {out_path}")


if __name__ == "__main__":
    main()
