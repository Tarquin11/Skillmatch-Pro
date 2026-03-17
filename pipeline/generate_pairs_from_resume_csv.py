from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

LIST_SPLIT_RE = re.compile(r"[,\n;/|]+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9+.#/\-\s]")

def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[\u00a0\t\r\n]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _normalize_skill(value: Any) -> str:
    text = _normalize_text(value).replace("&", " and ")
    text = NON_ALNUM_RE.sub(" ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if str(v).strip()]
        except Exception:
            pass
    return [p.strip() for p in LIST_SPLIT_RE.split(text) if p.strip()]


def _parse_skills(value: Any) -> list[str]:
    out: list[str] = []
    for item in _parse_list(value):
        skill = _normalize_skill(item)
        if skill:
            out.append(skill)
    # de-dupe while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


def _parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_experience_years(value: Any) -> float:
    text = str(value or "")
    match = re.search(r"(\d+(\.\d+)?)", text)
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except ValueError:
        return 0.0


def _hash_id(prefix: str, text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _overlap_ratio(
    candidate: list[str],
    required: list[str],
    fuzzy_threshold: float,
) -> tuple[float, int]:
    req = [r for r in required if r]
    cand = [c for c in candidate if c]
    if not req:
        return 0.0, 0
    cand_set = set(cand)
    overlap = 0
    for r in req:
        if r in cand_set:
            overlap += 1
            continue
        best = 0.0
        for c in cand_set:
            ratio = SequenceMatcher(None, r, c).ratio()
            if ratio > best:
                best = ratio
            if best >= fuzzy_threshold:
                break
        if best >= fuzzy_threshold:
            overlap += 1
    missing = max(0, len(req) - overlap)
    return overlap / len(req), missing


def _label_type(
    matched_score: float | None,
    overlap: float,
    missing: int,
    required_count: int,
    title_similarity: float,
    pos_score_min: float,
    pos_overlap_min: float,
    hard_score_min: float,
    hard_score_max: float,
    hard_overlap_min: float,
    easy_score_max: float,
    easy_overlap_max: float,
) -> str | None:
    if matched_score is not None:
        if matched_score >= pos_score_min and overlap >= pos_overlap_min and missing == 0:
            return "positive"
        if (
            hard_score_min <= matched_score < hard_score_max
            and missing >= 1
            and (overlap >= hard_overlap_min or (required_count <= 1 and title_similarity >= 0.6))
        ):
            return "hard_negative"
        if matched_score <= easy_score_max or overlap <= easy_overlap_max:
            return "easy_negative"
        return None

    # Fallback if score is missing
    if overlap >= pos_overlap_min and missing == 0:
        return "positive"
    if (overlap >= hard_overlap_min or (required_count <= 1 and title_similarity >= 0.6)) and missing >= 1:
        return "hard_negative"
    if overlap <= easy_overlap_max:
        return "easy_negative"
    return None


def _title_similarity(a: str, b: str) -> float:
    a_norm = _normalize_text(a)
    b_norm = _normalize_text(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def _extract_year(value: Any) -> str:
    text = str(value or "")
    match = re.search(r"(19|20)\d{2}", text)
    return match.group(0) if match else ""


def _pick_first(values: Iterable[str]) -> str:
    for v in values:
        if v:
            return v
    return ""

def _extract_year(value: Any) -> str:
    text = str(value or "")
    match = re.search(r"(19|20)\d{2}", text)
    return match.group(0) if match else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate positive + hard negative pairs from resume CSV")
    parser.add_argument("--input", default="data/raw/public/resume_data_for_ranking.csv")
    parser.add_argument("--out-csv", default="data/processed/ranking_pairs.csv")
    parser.add_argument("--out-jsonl", default="data/processed/ranking_pairs.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos-per-query", type=int, default=1)
    parser.add_argument("--hard-per-query", type=int, default=2)
    parser.add_argument("--easy-per-query", type=int, default=2)
    parser.add_argument("--pos-score-min", type=float, default=0.75)
    parser.add_argument("--pos-overlap-min", type=float, default=0.7)
    parser.add_argument("--hard-score-min", type=float, default=0.45)
    parser.add_argument("--hard-score-max", type=float, default=0.75)
    parser.add_argument("--hard-overlap-min", type=float, default=0.4)
    parser.add_argument("--easy-score-max", type=float, default=0.2)
    parser.add_argument("--easy-overlap-max", type=float, default=0.2)
    parser.add_argument("--skill-fuzzy-threshold", type=float, default=0.85)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_csv = Path(args.out_csv)
    out_jsonl = Path(args.out_jsonl)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    groups: dict[str, list[dict[str, Any]]] = {}
    candidate_pool: dict[str, dict[str, Any]] = {}
    job_pool: dict[str, dict[str, Any]] = {}

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            job_title = (row.get("job_position_name") or "").strip()
            job_skills_raw = row.get("skills_required") or row.get("related_skils_in_job") or ""
            job_skills = _parse_skills(job_skills_raw)

            time_key = _extract_year(
                row.get("start_dates") or row.get("passing_years") or row.get("issue_dates")
            )

            candidate_skills = _parse_skills(row.get("skills"))
            overlap, missing = _overlap_ratio(candidate_skills, job_skills, args.skill_fuzzy_threshold)
            required_count = len(job_skills)

            time_key =  _extract_year(row.get("start_dates") or row.get("passing_years") or row.get("issue_dates"))

            matched_score = _parse_float(row.get("matched_score"))
            title_similarity = _title_similarity(job_title, _pick_first(_parse_list(row.get("positions") or row.get("role_positions"))))
            label_type = _label_type(
                matched_score,
                overlap,
                missing,
                required_count,
                title_similarity,
                args.pos_score_min,
                args.pos_overlap_min,
                args.hard_score_min,
                args.hard_score_max,
                args.hard_overlap_min,
                args.easy_score_max,
                args.easy_overlap_max,
            )
            if label_type is None:
                continue

            job_key = "|".join(
                [
                    _normalize_text(job_title),
                    _normalize_text(job_skills_raw),
                    _normalize_text(row.get("educationaL_requirements")),
                    _normalize_text(row.get("experiencere_requirement")),
                ]
            )
            query_id = _hash_id("JOB", job_key)

            cand_key = "|".join(
                [
                    _normalize_text(row.get("career_objective")),
                    _normalize_text(row.get("skills")),
                    _normalize_text(row.get("educational_institution_name")),
                    _normalize_text(row.get("positions") or row.get("role_positions")),
                ]
            )
            candidate_id = _hash_id("CAND", cand_key or str(idx))

            position = _pick_first(_parse_list(row.get("positions") or row.get("role_positions")))

            employee = {
                "id": candidate_id,
                "full_name": "",
                "position": position,
                "department": "",
                "skills": candidate_skills,
                "summary": (row.get("career_objective") or "").strip(),
                "experience_years": 0.0,
            }
            job = {
                "id": query_id,
                "title": job_title,
                "description": (row.get("responsibilities.1") or row.get("responsibilities") or "").strip(),
                "required_skills": job_skills,
                "min_experience": _parse_experience_years(row.get("experiencere_requirement")),
            }

            record = {
                "query_id": query_id,
                "candidate_id": candidate_id,
                "job_position_name": job_title,
                "matched_score": matched_score if matched_score is not None else "",
                "overlap_ratio": round(overlap, 6),
                "title_similarity": round(title_similarity, 6),
                "time_key": time_key,
                "missing_required": missing,
                "label": 1 if label_type == "positive" else 0,
                "label_type": label_type,
                "employee": employee,
                "job": job,
                "time_key": time_key,
            }

            groups.setdefault(query_id, []).append(record)
            candidate_pool.setdefault(candidate_id, employee)
            job_pool.setdefault(query_id, job)

    selected: list[dict[str, Any]] = []
    for qid, records in groups.items():
        positives = [r for r in records if r["label_type"] == "positive"]
        hards = [r for r in records if r["label_type"] == "hard_negative"]
        easies = [r for r in records if r["label_type"] == "easy_negative"]

        if args.shuffle:
            rng.shuffle(positives)
            rng.shuffle(hards)
            rng.shuffle(easies)
        else:
            positives.sort(key=lambda r: (r["matched_score"], r["overlap_ratio"]), reverse=True)
            hards.sort(key=lambda r: (r["overlap_ratio"], r["matched_score"]), reverse=True)
            easies.sort(key=lambda r: (r["overlap_ratio"], r["matched_score"]))

        if not positives:
            continue

        picked = []
        picked.extend(positives[: args.pos_per_query])
        picked.extend(hards[: args.hard_per_query])
        picked.extend(easies[: args.easy_per_query])

        # remove duplicates by candidate_id within the query
        seen: set[str] = set()
        for rec in picked:
            cid = rec["candidate_id"]
            if cid in seen:
                continue
            seen.add(cid)
            selected.append(rec)

        # augment negatives from other candidates if needed
        need_hard = max(0, args.hard_per_query - len(hards))
        need_easy = max(0, args.easy_per_query - len(easies))
        if need_hard > 0 or need_easy > 0:
            job = job_pool.get(qid)
            if not job:
                continue
            required = job.get("required_skills") or []
            required_count = len(required)
            candidates: list[tuple[str, float, int, float, str]] = []
            for cid, cand in candidate_pool.items():
                if cid in seen:
                    continue
                overlap, missing = _overlap_ratio(
                    cand.get("skills") or [],
                    required,
                    args.skill_fuzzy_threshold,
                )
                title_similarity = _title_similarity(job.get("title") or "", cand.get("position") or "")
                label_type = _label_type(
                    None,
                    overlap,
                    missing,
                    required_count,
                    title_similarity,
                    args.pos_score_min,
                    args.pos_overlap_min,
                    args.hard_score_min,
                    args.hard_score_max,
                    args.hard_overlap_min,
                    args.easy_score_max,
                    args.easy_overlap_max,
                )
                if label_type in {"hard_negative", "easy_negative"}:
                    candidates.append((cid, overlap, missing, title_similarity, label_type))

            rng.shuffle(candidates)
            hard_candidates = sorted(
                [c for c in candidates if c[4] == "hard_negative"],
                key=lambda x: (x[1], x[3]),
                reverse=True,
            )
            easy_candidates = sorted(
                [c for c in candidates if c[4] == "easy_negative"],
                key=lambda x: (x[1], x[3]),
            )

            for cid, overlap, missing, title_similarity, _ in hard_candidates:
                if need_hard == 0:
                    break
                cand = candidate_pool[cid]
                selected.append(
                    {
                        "query_id": qid,
                        "candidate_id": cid,
                        "job_position_name": job.get("title") or "",
                        "matched_score": "",
                        "overlap_ratio": round(overlap, 6),
                        "title_similarity": round(title_similarity, 6),
                        "missing_required": missing,
                        "label": 0,
                        "label_type": "hard_negative",
                        "employee": cand,
                        "job": job,
                    }
                )
                seen.add(cid)
                need_hard -= 1

            for cid, overlap, missing, title_similarity, _ in easy_candidates:
                if need_easy == 0:
                    break
                cand = candidate_pool[cid]
                selected.append(
                    {
                        "query_id": qid,
                        "candidate_id": cid,
                        "job_position_name": job.get("title") or "",
                        "matched_score": "",
                        "overlap_ratio": round(overlap, 6),
                        "title_similarity": round(title_similarity, 6),
                        "missing_required": missing,
                        "label": 0,
                        "label_type": "easy_negative",
                        "employee": cand,
                        "job": job,
                    }
                )
                seen.add(cid)
                need_easy -= 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "candidate_id",
                "job_position_name",
                "matched_score",
                "overlap_ratio",
                "title_similarity",
                "missing_required",
                "label",
                "label_type",
            ],
        )
        writer.writeheader()
        for rec in selected:
            writer.writerow({k: rec.get(k, "") for k in writer.fieldnames})

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in selected:
            f.write(
                json.dumps(
                    {
                        "query_id": rec["query_id"],
                        "label": rec["label"],
                        "time_key": rec.get("time_key", ""),
                        "employee": rec["employee"],
                        "job": rec["job"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote {len(selected)} rows to {out_csv}")
    print(f"Wrote {len(selected)} pairs to {out_jsonl}")


if __name__ == "__main__":
    main()
