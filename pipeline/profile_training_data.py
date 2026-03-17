from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line :
                continue
            yield json.loads(line)

def main() -> None:
    parser = argparse.ArgumentParser(description="Profile training data for drift/imbalance")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="artifacts/data_profile.json")
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    total = 0
    pos = 0
    neg = 0
    missing_emp_skills = 0
    missing_job_skills = 0

    emp_skill_freq = Counter()
    job_skill_freq = Counter()

    for row in _iter_jsonl(Path(args.input)):
        total += 1
        label = int(row.get("label", 0))
        if label == 1:
            pos += 1
        else:
            neg += 1

        emp = row.get("employee") or {}
        job = row.get("job") or {}

        emp_skills = emp.get("skills") or []
        job_skills = job.get("required_skills") or []

        if not emp_skills:
            missing_emp_skills += 1
        if not job_skills:
            missing_job_skills += 1

        emp_skill_freq.update(emp_skills)
        job_skill_freq.update(job_skills)

    profile = {
        "total_rows": total,
        "positive": pos,
        "negative": neg,
        "pos_ratio": round(pos / total, 6) if total else 0.0,
        "missing_employee_skills_ratio": round(missing_emp_skills / total, 6) if total else 0.0,
        "missing_job_skills_ratio": round(missing_job_skills / total, 6) if total else 0.0,
        "top_employee_skills": emp_skill_freq.most_common(args.top_k),
        "top_job_skills": job_skill_freq.most_common(args.top_k),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()