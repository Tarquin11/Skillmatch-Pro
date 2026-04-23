from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from app.ai.skill_canonicalization import canonicalize_skill
from app.services.cv_parser import parse_cv_safe


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


TARGET_MIN_PRECISION_AT_1 = _env_float("CV_QUALITY_GATE_MIN_PRECISION_AT_1", 0.90)
TARGET_MIN_TECHNICAL_F1 = _env_float("CV_QUALITY_GATE_MIN_TECHNICAL_F1", 0.84)
TARGET_MAX_SEMANTIC_AUGMENT_FP_RATE = _env_float("CV_QUALITY_GATE_MAX_SEMANTIC_AUGMENT_FP_RATE", 0.08)
TARGET_MIN_BOARDS_EXCLUSIVITY_ACCURACY = _env_float("CV_QUALITY_GATE_MIN_BOARDS_EXCLUSIVITY_ACCURACY", 0.95)
TARGET_MAX_ECE = _env_float("CV_QUALITY_GATE_MAX_ECE", 0.06)


def _norm_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _norm_skill(value: str | None) -> str:
    return canonicalize_skill(value or "")


def _norm_board_item(value: str | None) -> str:
    return _norm_text(value)


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


def _known_skills_from_labels(labels: dict[str, Any]) -> list[str]:
    known = labels.get("known_skills")
    if isinstance(known, list):
        cleaned = [str(x).strip() for x in known if isinstance(x, str) and str(x).strip()]
        if cleaned:
            return cleaned

    values: list[str] = []
    for key in ("technical_skills", "skills", "tools"):
        raw = labels.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, str) and item.strip():
                values.append(item.strip())

    seen: set[str] = set()
    out: list[str] = []
    for skill in values:
        key = _norm_skill(skill)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(skill)
    return out


def _technical_rows(extracted_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in extracted_rows:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source", "")).lower()
        if source.startswith("softskill"):
            continue
        if not row.get("skill"):
            continue
        out.append(row)
    return out


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _ece(conf_points: list[tuple[float, int]], bins: int = 10) -> float | None:
    if not conf_points:
        return None
    n = len(conf_points)
    bin_counts = [0] * bins
    bin_conf_sum = [0.0] * bins
    bin_acc_sum = [0.0] * bins
    for conf, label in conf_points:
        c = max(0.0, min(1.0, float(conf)))
        idx = min(bins - 1, int(c * bins))
        bin_counts[idx] += 1
        bin_conf_sum[idx] += c
        bin_acc_sum[idx] += float(label)
    total = 0.0
    for i in range(bins):
        count = bin_counts[i]
        if count == 0:
            continue
        avg_conf = bin_conf_sum[i] / count
        avg_acc = bin_acc_sum[i] / count
        total += (count / n) * abs(avg_acc - avg_conf)
    return round(total, 6)


def _resolve_record_input(record: dict[str, Any], base_path: Path) -> tuple[bytes, str]:
    text = record.get("text")
    if isinstance(text, str):
        return text.encode("utf-8"), str(record.get("filename") or "snippet.txt")

    raw_path = record.get("path")
    if not raw_path:
        raise FileNotFoundError("record missing both 'text' and 'path'")
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = base_path / path
    if not path.exists():
        raise FileNotFoundError(f"record path does not exist: {path}")
    return path.read_bytes(), path.name


def _gate_min(value: float | None, target: float) -> dict[str, Any]:
    passed = value is not None and float(value) >= float(target)
    return {"value": value, "target_min": float(target), "passed": passed}


def _gate_max(value: float | None, target: float) -> dict[str, Any]:
    passed = value is not None and float(value) <= float(target)
    return {"value": value, "target_max": float(target), "passed": passed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed CV validation set and enforce extraction quality gates")
    parser.add_argument("--gold-jsonl", default="artifacts/gold/cv_quality_validation.jsonl")
    parser.add_argument("--out", default="artifacts/reports/cv_quality_gates_report.json")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--use-semantic", action="store_true")
    parser.add_argument("--use-hf-ner", action="store_true")
    parser.add_argument("--disable-semantic-augment", action="store_true")
    parser.add_argument("--include-per-item", action="store_true")
    args = parser.parse_args()

    gold_path = Path(args.gold_jsonl).expanduser()
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold JSONL not found: {gold_path}")
    records = _load_jsonl(gold_path)

    tp = fp = fn = 0
    top1_total = 0
    top1_correct = 0
    semantic_aug_total = 0
    semantic_aug_fp = 0
    board_total = 0
    board_correct = 0
    ece_points: list[tuple[float, int]] = []
    parse_errors = 0
    per_item: list[dict[str, Any]] = []

    for rec in records:
        labels = rec.get("labels", {})
        if not isinstance(labels, dict):
            labels = {}
        known_skills = _known_skills_from_labels(labels)
        gold_tech = {
            _norm_skill(str(item))
            for item in (labels.get("technical_skills") or labels.get("skills") or [])
            if str(item).strip()
        }
        gold_tech.discard("")

        try:
            file_bytes, filename = _resolve_record_input(rec, gold_path.parent)
            parsed = parse_cv_safe(
                file_bytes=file_bytes,
                filename=filename,
                known_skills=known_skills,
                min_confidence=float(args.min_confidence),
                use_semantic=bool(args.use_semantic),
                use_hf_ner=bool(args.use_hf_ner),
                use_semantic_augment=not bool(args.disable_semantic_augment),
            )
        except Exception as exc:
            parse_errors += 1
            if args.include_per_item:
                per_item.append(
                    {
                        "id": rec.get("id"),
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            continue

        extracted = parsed.get("extracted_skills") or []
        if not isinstance(extracted, list):
            extracted = []
        tech_rows = _technical_rows(extracted)
        pred_tech = {
            _norm_skill(str(row.get("skill", "")))
            for row in tech_rows
            if _norm_skill(str(row.get("skill", "")))
        }

        tp += len(pred_tech & gold_tech)
        fp += len(pred_tech - gold_tech)
        fn += len(gold_tech - pred_tech)

        if gold_tech:
            top1_total += 1
            if tech_rows:
                top_row = max(
                    tech_rows,
                    key=lambda row: float(row.get("confidence", 0.0))
                    if isinstance(row.get("confidence", 0.0), (int, float))
                    else 0.0,
                )
                top_skill = _norm_skill(str(top_row.get("skill", "")))
                if top_skill and top_skill in gold_tech:
                    top1_correct += 1

        for row in tech_rows:
            source = str(row.get("source", "")).lower()
            skill = _norm_skill(str(row.get("skill", "")))
            if source.startswith("semantic_augment"):
                semantic_aug_total += 1
                if not skill or skill not in gold_tech:
                    semantic_aug_fp += 1
            try:
                conf = float(row.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            ece_points.append((max(0.0, min(1.0, conf)), 1 if skill in gold_tech else 0))

        pred_certs = {
            _norm_board_item(str(item))
            for item in (parsed.get("certifications") or [])
            if _norm_board_item(str(item))
        }
        pred_projects = {
            _norm_board_item(str(item))
            for item in (parsed.get("hands_on_projects") or [])
            if _norm_board_item(str(item))
        }
        gold_certs = {
            _norm_board_item(str(item))
            for item in (labels.get("certifications") or [])
            if _norm_board_item(str(item))
        }
        gold_projects = {
            _norm_board_item(str(item))
            for item in (labels.get("hands_on_projects") or [])
            if _norm_board_item(str(item))
        }

        if gold_certs or gold_projects:
            for cert in gold_certs:
                board_total += 1
                if cert in pred_certs and cert not in pred_projects:
                    board_correct += 1
            for project in gold_projects:
                board_total += 1
                if project in pred_projects and project not in pred_certs:
                    board_correct += 1
        else:
            board_total += 1
            if not (pred_certs & pred_projects):
                board_correct += 1

        if args.include_per_item:
            per_item.append(
                {
                    "id": rec.get("id"),
                    "ok": bool(parsed.get("ok", True)),
                    "gold_technical_skills": sorted(gold_tech),
                    "pred_technical_skills": sorted(pred_tech),
                    "pred_semantic_augment": sorted(
                        {
                            _norm_skill(str(row.get("skill", "")))
                            for row in tech_rows
                            if str(row.get("source", "")).lower().startswith("semantic_augment")
                        }
                    ),
                }
            )

    technical_prf = _prf(tp, fp, fn)
    precision_at_1 = round(top1_correct / top1_total, 6) if top1_total else None
    semantic_aug_fp_rate = round(semantic_aug_fp / semantic_aug_total, 6) if semantic_aug_total else 0.0
    boards_accuracy = round(board_correct / board_total, 6) if board_total else None
    ece_value = _ece(ece_points, bins=10)

    gates = {
        "precision_at_1": _gate_min(precision_at_1, TARGET_MIN_PRECISION_AT_1),
        "technical_f1": _gate_min(technical_prf["f1"], TARGET_MIN_TECHNICAL_F1),
        "semantic_augment_fp_rate": _gate_max(semantic_aug_fp_rate, TARGET_MAX_SEMANTIC_AUGMENT_FP_RATE),
        "boards_mutual_exclusivity_accuracy": _gate_min(
            boards_accuracy, TARGET_MIN_BOARDS_EXCLUSIVITY_ACCURACY
        ),
        "ece": _gate_max(ece_value, TARGET_MAX_ECE),
    }
    passed = all(bool(item.get("passed", False)) for item in gates.values())

    report: dict[str, Any] = {
        "config": {
            "records": len(records),
            "gold_jsonl": str(gold_path),
            "min_confidence": float(args.min_confidence),
            "use_semantic": bool(args.use_semantic),
            "use_hf_ner": bool(args.use_hf_ner),
            "use_semantic_augment": not bool(args.disable_semantic_augment),
        },
        "metrics": {
            "technical_skills": {
                "precision_at_1": precision_at_1,
                "precision": technical_prf["precision"],
                "recall": technical_prf["recall"],
                "f1": technical_prf["f1"],
            },
            "semantic_only": {
                "semantic_augment_fp_rate": semantic_aug_fp_rate,
            },
            "boards_quality": {
                "mutual_exclusivity_accuracy": boards_accuracy,
            },
            "calibration": {
                "ece": ece_value,
            },
        },
        "counts": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "top1_total": top1_total,
            "top1_correct": top1_correct,
            "semantic_augment_total": semantic_aug_total,
            "semantic_augment_fp": semantic_aug_fp,
            "boards_total": board_total,
            "boards_correct": board_correct,
            "ece_points": len(ece_points),
            "parse_errors": parse_errors,
        },
        "gates": gates,
        "passed": passed,
    }
    if args.include_per_item:
        report["per_item"] = per_item

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report to: {out_path}")


if __name__ == "__main__":
    main()
