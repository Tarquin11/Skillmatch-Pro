from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

SKILL_SPLIT_RE = re.compile(r"[,\|;/]+")
NON_ALNUM_SKILL_RE = re.compile(r"[^a-z0-9+.#/\-\s]")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _data_dir() -> Path:
    return _repo_root() / "data"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[\u00a0\t\r\n]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _normalize_skill(value: Any) -> str:
    text = _normalize_text(value).replace("&", " and ")
    text = NON_ALNUM_SKILL_RE.sub(" ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _dedupe_preserve(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in values:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _parse_skills(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = SKILL_SPLIT_RE.split(raw)
        return _dedupe_preserve(_normalize_skill(p) for p in parts if _normalize_skill(p))
    if isinstance(raw, dict):
        name = raw.get("name") or raw.get("skill") or raw.get("label") or ""
        val = _normalize_skill(name)
        return [val] if val else []
    if isinstance(raw, (list, tuple, set)):
        out: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                name = item.get("name") or item.get("skill") or item.get("label") or ""
                if isinstance(name, dict):
                    name = name.get("name", "")
            else:
                name = item
            val = _normalize_skill(name)
            if val:
                out.append(val)
        return _dedupe_preserve(out)
    return []


def _iter_pairs(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "pairs" in data:
            data = data["pairs"]
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    yield row
            return
        raise ValueError("JSON input must be a list or {'pairs': [...]} object.")

    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield dict(row)
        return

    raise ValueError("Unsupported input format. Use .jsonl, .json, or .csv")


def _write_jsonl(rows: Iterable[dict[str, Any]], out_path: Path) -> int:
    _ensure_parent(out_path)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def ingest_raw(input_path: Path, raw_out: Path) -> int:
    return _write_jsonl(_iter_pairs(input_path), raw_out)


def _clean_pair(row: dict[str, Any], idx: int) -> dict[str, Any] | None:
    employee = row.get("employee") or {}
    job = row.get("job") or {}
    if not isinstance(employee, dict) or not isinstance(job, dict):
        return None

    try:
        label = int(row.get("label"))
    except (TypeError, ValueError):
        return None
    if label not in (0, 1):
        return None

    emp_skills = _parse_skills(
        employee.get("skills")
        or employee.get("skill")
        or employee.get("competences")
        or employee.get("competencies")
    )
    job_skills = _parse_skills(
        job.get("required_skills")
        or job.get("skills")
        or job.get("must_have_skills")
        or job.get("nice_to_have_skills")
    )

    query_id = row.get("query_id") or row.get("job_id") or job.get("id") or idx
    employee_id = employee.get("id") or employee.get("employee_id") or idx

    return {
        "query_id": query_id,
        "label": label,
        "employee": {
            "id": employee_id,
            "full_name": _normalize_text(
                employee.get("full_name")
                or f"{employee.get('first_name', '')} {employee.get('last_name', '')}".strip()
            ),
            "position": _normalize_text(employee.get("position") or employee.get("title")),
            "department": _normalize_text(
                employee.get("department") or employee.get("departement") or employee.get("dept")
            ),
            "skills": emp_skills,
            "summary": _normalize_text(employee.get("summary") or employee.get("resume_summary")),
            "experience_years": float(
                employee.get("experience_years") or employee.get("years_experience") or 0
            ),
        },
        "job": {
            "id": job.get("id") or row.get("job_id"),
            "title": _normalize_text(job.get("title") or job.get("job_title")),
            "description": _normalize_text(job.get("description")),
            "department": _normalize_text(job.get("department") or job.get("departement") or job.get("dept")),
            "required_skills": job_skills,
            "min_experience": float(
                job.get("min_experience") or job.get("required_experience_years") or 0
            ),
        },
    }


def clean_pairs(raw_path: Path, clean_out: Path) -> int:
    def _iter_clean() -> Iterator[dict[str, Any]]:
        for idx, row in enumerate(_iter_pairs(raw_path)):
            cleaned = _clean_pair(row, idx)
            if cleaned:
                yield cleaned

    return _write_jsonl(_iter_clean(), clean_out)


def add_features(clean_path: Path, features_out: Path) -> int:
    def _iter_features() -> Iterator[dict[str, Any]]:
        for row in _iter_pairs(clean_path):
            employee = row.get("employee") or {}
            job = row.get("job") or {}
            emp_skills = set(employee.get("skills") or [])
            job_skills = set(job.get("required_skills") or [])
            overlap = emp_skills & job_skills
            req_count = len(job_skills)
            overlap_ratio = (len(overlap) / req_count) if req_count else 0.0

            row["features"] = {
                "employee_skill_count": len(emp_skills),
                "required_skill_count": req_count,
                "skill_overlap_count": len(overlap),
                "skill_overlap_ratio": round(overlap_ratio, 6),
            }
            yield row

    return _write_jsonl(_iter_features(), features_out)


def _split_bucket(value: Any, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % 10_000
    train_cutoff = int(train_ratio * 10_000)
    val_cutoff = int((train_ratio + val_ratio) * 10_000)
    if bucket < train_cutoff:
        return "train"
    if bucket < val_cutoff:
        return "val"
    return "test"

def _get_group_key(row: dict[str, Any], group_by: str | None) -> Any:
    if not group_by:
        return None
    key = group_by.lower()
    employee = row.get("employee")
    if not isinstance(employee, dict):
        employee = {}
    job = row.get("job")
    if not isinstance(job, dict):
        job = {}
    if key in {"candidate_id", "employee_id"}:
        return employee.get("id") or row.get("candidate_id") or row.get("employee_id")
    if key == "query_id":
        return row.get("query_id") or row.get("job_id") or job.get("id")
    if key == "job_id":
        return job.get("id") or row.get("job_id") or row.get("query_id")
    return row.get(group_by)

def split_pairs(
    features_path: Path,
    splits_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    group_by: str | None,
    seed: int,
) -> dict[str, int]:
    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    splits_dir.mkdir(parents=True, exist_ok=True)
    out_files = {
        "train": (splits_dir / "train.jsonl").open("w", encoding="utf-8"),
        "val": (splits_dir / "val.jsonl").open("w", encoding="utf-8"),
        "test": (splits_dir / "test.jsonl").open("w", encoding="utf-8"),
    }
    counts = {"train": 0, "val": 0, "test": 0}
    rng = random.Random(seed)

    try:
        for idx, row in enumerate(_iter_pairs(features_path)):
            key = _get_group_key(row, group_by)
            if key is None:
                roll = rng.random()
                if roll < train_ratio:
                    split = "train"
                elif roll < train_ratio + val_ratio:
                    split = "val"
                else:
                    split = "test"
            else:
                split = _split_bucket(key, train_ratio, val_ratio)
            out_files[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for f in out_files.values():
            f.close()

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline data pipeline (raw -> clean -> features -> splits)")
    parser.add_argument("--input", required=True, help="Raw input path (.jsonl/.json/.csv)")
    parser.add_argument("--raw-out", default=str(_data_dir() / "raw" / "pairs.jsonl"))
    parser.add_argument("--clean-out", default=str(_data_dir() / "clean" / "pairs.jsonl"))
    parser.add_argument("--features-out", default=str(_data_dir() / "features" / "pairs.jsonl"))
    parser.add_argument("--splits-dir", default=str(_data_dir() / "splits"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--group-by", default="query_id", help="query_id, job_id, candidate_id, employee_id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    raw_out = Path(args.raw_out)
    clean_out = Path(args.clean_out)
    features_out = Path(args.features_out)
    splits_dir = Path(args.splits_dir)

    if not args.skip_ingest:
        count = ingest_raw(input_path, raw_out)
        print(f"Ingested {count} raw rows -> {raw_out}")

    if not args.skip_clean:
        count = clean_pairs(raw_out, clean_out)
        print(f"Cleaned {count} rows -> {clean_out}")

    if not args.skip_features:
        count = add_features(clean_out, features_out)
        print(f"Feature-ready rows {count} -> {features_out}")

    if not args.skip_split:
        counts = split_pairs(
            features_out,
            splits_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            group_by=args.group_by or None,
            seed=args.seed,
        )
        print(f"Split counts: {counts}")
        print(f"Splits written to {splits_dir}")


if __name__ == "__main__":
    main()
