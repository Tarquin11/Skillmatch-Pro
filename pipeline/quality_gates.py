#Minimum Data Quality Gates
from __future__ import annotations
import argparse
import json
from pathlib import Path

def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimum data quality gates")
    parser.add_argument("--input", required=True)
    parser.add_argument("--min-rows", type=int, default=1000)
    parser.add_argument("--min-pos-ratio", type=float, default=0.1)
    parser.add_argument("--max-missing-emp-skills", type=float, default=0.3)
    parser.add_argument("--max-missing-job-skills", type=float, default=0.3)
    args = parser.parse_args()

    total = 0
    pos = 0
    missing_emp = 0
    missing_job = 0

    for row in _iter_jsonl(Path(args.input)):
        total += 1
        if int(row.get("label", 0)) == 1:
            pos += 1
        emp = row.get("employee") or {}
        job = row.get("job") or {}
        if not emp.get("skills"):
            missing_emp += 1
        if not job.get("required_skills"):
            missing_job += 1

    if total < args.min_rows:
        raise SystemExit(f"FAIL: rows={total} < min_rows={args.min_rows}")

    pos_ratio = pos / total if total else 0
    if pos_ratio < args.min_pos_ratio:
        raise SystemExit(f"FAIL: pos_ratio={pos_ratio:.3f} < min_pos_ratio={args.min_pos_ratio}")

    if (missing_emp / total) > args.max_missing_emp_skills:
        raise SystemExit("FAIL: too many rows missing employee skills")

    if (missing_job / total) > args.max_missing_job_skills:
        raise SystemExit("FAIL: too many rows missing job skills")

    print("PASS")

if __name__ == "__main__":
    main()
