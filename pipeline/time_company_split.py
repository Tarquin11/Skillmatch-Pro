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

def _get_company(row: dict[str, Any]) -> str:
    emp = row.get("employee") or {}
    job = row.get("job") or {}
    return (
        str(emp.get("company") or emp.get("employer") or job.get("company") or job.get("employer") or "")
        .strip()
        .lower()
    )

def _get_time_key(row: dict[str, Any], field:str) -> str:
    if field in row:
        return str(row.get(field) or "")
    emp = row.get("employee") or {}
    job = row.get("job") or {}
    return str(emp.get(field) or job.get(field) or "")

def main() -> None:
    parser = argparse.ArgumentParser(description="Split JSONL by time + company")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-out", default="data/splits/train.jsonl")
    parser.add_argument("--val-out", default="data/splits/val.jsonl")
    parser.add_argument("--test-out", default="data/splits/test.jsonl")
    parser.add_argument("--time-field", default="job_posted_at")
    parser.add_argument("--train-cutoff", required=True, help="e.g. 2024-01-01")
    parser.add_argument("--val-cutoff", required=True, help="e.g. 2024-06-01")
    parser.add_argument("--company-holdout", action="store_true")
    parser.add_argument("--company-list", default="data/registry/company_holdout.txt")
    args = parser.parse_args()

    train_path = Path(args.train_out)
    val_path = Path(args.val_out)
    test_path = Path(args.test_out)
    train_path.parent.mkdir(parents=True, exist_ok=True)

    holdout_companies = set()
    if args.company_holdout:
        company_file = Path(args.company_list)
        if company_file.exists():
            holdout_companies = {c.strip().lower() for c in company_file.read_text().splitlines() if c.strip()}

    train_f = train_path.open("w", encoding="utf-8")
    val_f = val_path.open("w", encoding="utf-8")
    test_f = test_path.open("w", encoding="utf-8")

    train_cut = args.train_cutoff
    val_cut = args.val_cutoff

    try:
        for row in _iter_jsonl(Path(args.input)):
            company = _get_company(row)
            time_val = _get_time_key(row, args.time_field)

            if args.company_holdout and company in holdout_companies:
                test_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            if time_val and time_val < train_cut:
                train_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            elif time_val and time_val < val_cut:
                val_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                test_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        train_f.close()
        val_f.close()
        test_f.close()

    print(f"Split done: {train_path}, {val_path}, {test_path}")


if __name__ == "__main__":
    main()