from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from app.ai.skill_canonicalization import canonicalize_skill
from app.services.cv_parser import (
    detect_experience_years,
    detect_skill_spans_with_ensemble,
    detect_skills_with_confidence,
    detect_title,
    extract_language_details_from_sections,
    extract_open_vocabulary_skill_rows,
)


def _norm_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _norm_skill(value: str | None) -> str:
    return canonicalize_skill(value or "")


def _norm_lang(value: str | None) -> str:
    return _norm_text(value)


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _gold_langs(labels: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for item in labels.get("languages", []) or []:
        if isinstance(item, dict):
            name = _norm_lang(str(item.get("language", "")))
        else:
            name = _norm_lang(str(item))
        if name:
            out.add(name)
    return out


def _gold_lang_levels(labels: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in labels.get("languages", []) or []:
        if not isinstance(item, dict):
            continue
        name = _norm_lang(str(item.get("language", "")))
        level = _norm_text(str(item.get("level", ""))).upper()
        if name and level:
            out[name] = level
    return out


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
    except Exception:
        return None
    if not (f == f):  # NaN guard
        return None
    return f


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CV snippet extraction quality (skills/tools/languages/title/experience)")
    parser.add_argument("--gold-jsonl", default="artifacts/gold/cv_snippets_gold.jsonl")
    parser.add_argument("--out", default="artifacts/reports/cv_eval_report.json")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--use-semantic", action="store_true")
    args = parser.parse_args()

    gold_path = Path(args.gold_jsonl).expanduser()
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold JSONL not found: {gold_path}")
    records = _load_jsonl(gold_path)

    totals = {
        "skills": {"tp": 0, "fp": 0, "fn": 0},
        "tools": {"tp": 0, "fp": 0, "fn": 0},
        "languages": {"tp": 0, "fp": 0, "fn": 0},
    }
    title_total = 0
    title_ok = 0
    exp_total = 0
    exp_errors: list[float] = []
    level_total = 0
    level_ok = 0
    project_fp_num = 0
    project_fp_den = 0
    per_item: list[dict[str, Any]] = []

    for rec in records:
        text = str(rec.get("text", ""))
        labels = rec.get("labels", {}) if isinstance(rec.get("labels"), dict) else {}
        known_skills = list({*(labels.get("skills", []) or []), *(labels.get("tools", []) or [])})
        catalog_rows = detect_skills_with_confidence(
            text=text,
            known_skills=known_skills,
            min_confidence=float(args.min_confidence),
            use_semantic=bool(args.use_semantic),
        )
        catalog_keys = {_norm_skill(str(r.get("skill", ""))) for r in catalog_rows if isinstance(r, dict)}
        rejected_project: list[str] = []
        open_rows = extract_open_vocabulary_skill_rows(
            text=text,
            catalog_skill_keys={k for k in catalog_keys if k},
            min_confidence=float(args.min_confidence),
            rejected_out=rejected_project,
        )
        ner_rows = detect_skill_spans_with_ensemble(
            text=text,
            known_skills=known_skills,
            min_confidence=float(args.min_confidence),
        )
        language_details = extract_language_details_from_sections(text)

        pred_catalog = {_norm_skill(str(r.get("skill", ""))) for r in catalog_rows + ner_rows if isinstance(r, dict)}
        pred_open = {_norm_skill(str(r.get("skill", ""))) for r in open_rows if isinstance(r, dict)}
        pred_all = pred_catalog | pred_open
        pred_langs = {_norm_lang(str(item.get("language", ""))) for item in language_details if isinstance(item, dict)}
        pred_project = {_norm_text(s) for s in rejected_project if s}

        gold_skills = {_norm_skill(s) for s in (labels.get("skills", []) or []) if s}
        gold_tools = {_norm_skill(s) for s in (labels.get("tools", []) or []) if s}
        gold_langs = _gold_langs(labels)
        gold_project = {_norm_text(s) for s in (labels.get("project_text", []) or []) if s}

        pred_skill_only = {s for s in pred_all if s and s not in gold_tools}
        pred_tool_only = {s for s in pred_all if s and s in gold_tools}

        for key, pred, gold in (
            ("skills", pred_skill_only, gold_skills),
            ("tools", pred_tool_only, gold_tools),
            ("languages", pred_langs, gold_langs),
        ):
            tp = len(pred & gold)
            fp = len(pred - gold)
            fn = len(gold - pred)
            totals[key]["tp"] += tp
            totals[key]["fp"] += fp
            totals[key]["fn"] += fn

        # project-text false positives
        project_fp_num += len({s for s in pred_all if s and s in gold_project})
        project_fp_den += len(pred_all)

        expected_title = labels.get("title")
        predicted_title = detect_title(text)
        title_match = None
        if expected_title is not None:
            title_total += 1
            a = _norm_text(str(expected_title))
            b = _norm_text(str(predicted_title or ""))
            title_match = bool(a and b and (a in b or b in a))
            title_ok += int(title_match)

        expected_exp = labels.get("experience_years")
        predicted_exp = detect_experience_years(text)
        if expected_exp is not None:
            exp_total += 1
            e = _safe_float(expected_exp)
            p = _safe_float(predicted_exp)
            if e is not None and p is not None:
                exp_errors.append(abs(p - e))

        gold_levels = _gold_lang_levels(labels)
        pred_levels: dict[str, str] = {}
        for item in language_details:
            if not isinstance(item, dict):
                continue
            name = _norm_lang(str(item.get("language", "")))
            level = _norm_text(str(item.get("level", ""))).upper()
            if name and level:
                pred_levels[name] = level
        for lang, level in gold_levels.items():
            level_total += 1
            if pred_levels.get(lang) == level:
                level_ok += 1

        per_item.append(
            {
                "id": rec.get("id"),
                "pred": {
                    "skills_or_tools": sorted(pred_all),
                    "languages": sorted(pred_langs),
                    "project_text": sorted(pred_project),
                    "title": predicted_title,
                    "experience_years": predicted_exp,
                },
                "gold": labels,
            }
        )

    report = {
        "config": {
            "records": len(records),
            "gold_jsonl": str(gold_path),
            "min_confidence": float(args.min_confidence),
            "use_semantic": bool(args.use_semantic),
        },
        "metrics": {
            "skills": _prf(totals["skills"]["tp"], totals["skills"]["fp"], totals["skills"]["fn"]),
            "tools": _prf(totals["tools"]["tp"], totals["tools"]["fp"], totals["tools"]["fn"]),
            "languages": _prf(totals["languages"]["tp"], totals["languages"]["fp"], totals["languages"]["fn"]),
            "language_level_accuracy": round(level_ok / level_total, 6) if level_total else None,
            "title_accuracy": round(title_ok / title_total, 6) if title_total else None,
            "experience_mae": round(sum(exp_errors) / len(exp_errors), 6) if exp_errors else None,
            "project_text_fp_rate": round(project_fp_num / project_fp_den, 6) if project_fp_den else 0.0,
        },
        "counts": totals,
        "per_item": per_item,
    }

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report to: {out_path}")


if __name__ == "__main__":
    main()
