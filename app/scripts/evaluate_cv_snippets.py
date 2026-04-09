from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from app.ai.skill_canonicalization import canonicalize_skill
from app.services.cv_parser import parse_cv_safe

CHANNEL_KEYS = (
    "catalog_match",
    "open_vocab",
    "soft_skill",
    "sentence",
    "semantic_augment",
    "ner_span",
    "other",
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
        raw_level = item.get("level")
        if raw_level is None:
            continue
        level = _norm_text(str(raw_level)).upper()
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


def _bucket_from_source(source: str) -> str:
    """Map extraction row source to a stable channel (for per-channel metrics)."""
    s = (source or "").strip().lower()
    if s.startswith("cv_section:"):
        return "open_vocab"
    if s.startswith("ner_span"):
        return "ner_span"
    if s.startswith("softskill"):
        return "soft_skill"
    if s.startswith("sentence_"):
        return "sentence"
    if s.startswith("semantic_augment"):
        return "semantic_augment"
    if s.startswith("exact:") or s.startswith("fuzzy:") or s == "synonym" or s.startswith("semantic:"):
        return "catalog_match"
    if s == "legacy":
        return "catalog_match"
    return "other"


def _pred_skills_by_channel(extracted_skills: list[Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {k: set() for k in CHANNEL_KEYS}
    for r in extracted_skills:
        if not isinstance(r, dict):
            continue
        sk = _norm_skill(str(r.get("skill", "")))
        if not sk:
            continue
        ch = _bucket_from_source(str(r.get("source", "")))
        if ch not in out:
            ch = "other"
        out[ch].add(sk)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CV snippet extraction via parse_cv_safe (skills by channel, languages, title, experience)"
    )
    parser.add_argument("--gold-jsonl", default="artifacts/gold/cv_snippets_gold.jsonl")
    parser.add_argument("--out", default="artifacts/reports/cv_eval_report.json")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--use-semantic", action="store_true", help="Lexical catalog path semantic merge (detect_skills)")
    parser.add_argument("--use-hf-ner", action="store_true")
    parser.add_argument("--use-semantic-augment", action="store_true")
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
    channel_totals: dict[str, dict[str, int]] = {k: {"tp": 0, "fp": 0, "fn": 0} for k in CHANNEL_KEYS}
    channel_items_evaluated: dict[str, int] = {k: 0 for k in CHANNEL_KEYS}

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
        ks_override = labels.get("known_skills")
        if isinstance(ks_override, list) and ks_override:
            known_skills = [str(s).strip() for s in ks_override if isinstance(s, str) and str(s).strip()]
        else:
            known_skills = list(
                dict.fromkeys(
                    [*(labels.get("skills", []) or []), *(labels.get("tools", []) or [])]
                )
            )
            known_skills = [str(s).strip() for s in known_skills if isinstance(s, str) and str(s).strip()]

        file_bytes = text.encode("utf-8")
        parsed = parse_cv_safe(
            file_bytes=file_bytes,
            filename="snippet.txt",
            known_skills=known_skills,
            min_confidence=float(args.min_confidence),
            use_semantic=bool(args.use_semantic),
            use_hf_ner=bool(args.use_hf_ner),
            use_semantic_augment=bool(args.use_semantic_augment),
        )

        extracted = parsed.get("extracted_skills") or []
        if not isinstance(extracted, list):
            extracted = []

        pred_all = {_norm_skill(str(r.get("skill", ""))) for r in extracted if isinstance(r, dict) and r.get("skill")}
        pred_all.discard("")

        language_details = parsed.get("language_details") or []
        if not isinstance(language_details, list):
            language_details = []

        pred_langs = {
            _norm_lang(str(item.get("language", "")))
            for item in language_details
            if isinstance(item, dict) and item.get("language")
        }
        pred_langs.discard("")

        pred_project = set()
        ch_proj = parsed.get("extraction_channels", {})
        if isinstance(ch_proj, dict):
            pt = ch_proj.get("project_text") or []
            if isinstance(pt, list):
                pred_project = {_norm_text(str(s)) for s in pt if s}

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

        project_fp_num += len({s for s in pred_all if s and s in gold_project})
        project_fp_den += len(pred_all)

        sbc = labels.get("skills_by_channel")
        pred_by_ch = _pred_skills_by_channel(extracted)
        if isinstance(sbc, dict):
            for ch in CHANNEL_KEYS:
                raw_g = sbc.get(ch)
                if not isinstance(raw_g, list) or not raw_g:
                    continue
                gold_ch = {_norm_skill(str(x)) for x in raw_g if x}
                gold_ch.discard("")
                if not gold_ch:
                    continue
                pred_ch = pred_by_ch.get(ch, set())
                tp = len(pred_ch & gold_ch)
                fp = len(pred_ch - gold_ch)
                fn = len(gold_ch - pred_ch)
                channel_totals[ch]["tp"] += tp
                channel_totals[ch]["fp"] += fp
                channel_totals[ch]["fn"] += fn
                channel_items_evaluated[ch] += 1

        expected_title = labels.get("title")
        predicted_title = parsed.get("predicted_title")
        title_match = None
        if expected_title is not None:
            title_total += 1
            a = _norm_text(str(expected_title))
            b = _norm_text(str(predicted_title or ""))
            title_match = bool(a and b and (a in b or b in a))
            title_ok += int(title_match)

        expected_exp = labels.get("experience_years")
        predicted_exp = parsed.get("predicted_experience_years")
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
                    "skills_by_channel": {k: sorted(v) for k, v in pred_by_ch.items() if v},
                    "languages": sorted(pred_langs),
                    "project_text": sorted(pred_project),
                    "title": predicted_title,
                    "experience_years": predicted_exp,
                },
                "gold": labels,
            }
        )

    channel_metrics: dict[str, Any] = {}
    for ch in CHANNEL_KEYS:
        t = channel_totals[ch]
        if t["tp"] + t["fp"] + t["fn"] == 0:
            channel_metrics[ch] = None
        else:
            channel_metrics[ch] = {
                **_prf(t["tp"], t["fp"], t["fn"]),
                "gold_spans_evaluated": channel_items_evaluated[ch],
            }

    report = {
        "config": {
            "records": len(records),
            "gold_jsonl": str(gold_path),
            "min_confidence": float(args.min_confidence),
            "use_semantic": bool(args.use_semantic),
            "use_hf_ner": bool(args.use_hf_ner),
            "use_semantic_augment": bool(args.use_semantic_augment),
            "parser": "parse_cv_safe",
        },
        "metrics": {
            "skills": _prf(totals["skills"]["tp"], totals["skills"]["fp"], totals["skills"]["fn"]),
            "tools": _prf(totals["tools"]["tp"], totals["tools"]["fp"], totals["tools"]["fn"]),
            "languages": _prf(totals["languages"]["tp"], totals["languages"]["fp"], totals["languages"]["fn"]),
            "language_level_accuracy": round(level_ok / level_total, 6) if level_total else None,
            "title_accuracy": round(title_ok / title_total, 6) if title_total else None,
            "experience_mae": round(sum(exp_errors) / len(exp_errors), 6) if exp_errors else None,
            "project_text_fp_rate": round(project_fp_num / project_fp_den, 6) if project_fp_den else 0.0,
            "channels": channel_metrics,
        },
        "counts": {
            "skills_tools_langs": totals,
            "channels": channel_totals,
            "channel_gold_spans": channel_items_evaluated,
        },
        "per_item": per_item,
    }

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report to: {out_path}")


if __name__ == "__main__":
    main()
