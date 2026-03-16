from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any

def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _get_candidate_id(row: dict[str, Any]) -> str:
    emp = row.get("employee") or {}
    return str(emp.get("id") or "")


def _get_job_id(row: dict[str, Any]) -> str:
    job = row.get("job") or {}
    return str(job.get("id") or row.get("query_id") or "")


def _get_pair_key(row: dict[str, Any]) -> str:
    return f"{_get_job_id(row)}::{_get_candidate_id(row)}"


def _load_split(path: Path):
    candidates = set()
    jobs = set()
    pairs = set()

    for row in _iter_jsonl(path):
        cid = _get_candidate_id(row)
        jid = _get_job_id(row)
        if cid:
            candidates.add(cid)
        if jid:
            jobs.add(jid)
        if cid and jid:
            pairs.add(f"{jid}::{cid}")
    return candidates, jobs, pairs


def _overlap(a: set, b: set) -> int:
    return len(a & b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check train/val/test split leakage")
    parser.add_argument("--splits-dir", default="data/splits")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    train_path = splits_dir / "train.jsonl"
    val_path = splits_dir / "val.jsonl"
    test_path = splits_dir / "test.jsonl"

    missing = [p for p in [train_path, val_path, test_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing split files: {missing}")

    train_c, train_j, train_p = _load_split(train_path)
    val_c, val_j, val_p = _load_split(val_path)
    test_c, test_j, test_p = _load_split(test_path)

    report = {
        "train_count": len(train_p),
        "val_count": len(val_p),
        "test_count": len(test_p),
        "candidate_overlap": {
            "train_val": _overlap(train_c, val_c),
            "train_test": _overlap(train_c, test_c),
            "val_test": _overlap(val_c, test_c),
        },
        "job_overlap": {
            "train_val": _overlap(train_j, val_j),
            "train_test": _overlap(train_j, test_j),
            "val_test": _overlap(val_j, test_j),
        },
        "pair_overlap": {
            "train_val": _overlap(train_p, val_p),
            "train_test": _overlap(train_p, test_p),
            "val_test": _overlap(val_p, test_p),
        },
    }

    print(json.dumps(report, indent=2))

    leakage = any(v > 0 for v in report["candidate_overlap"].values()) \
        or any(v > 0 for v in report["job_overlap"].values()) \
        or any(v > 0 for v in report["pair_overlap"].values())

    if leakage:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
