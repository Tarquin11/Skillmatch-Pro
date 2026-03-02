import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


def to_list(v):
    if v is None:
        return []
    if isinstance(v, np.ndarray):
        return [str(x) for x in v.tolist()]
    if isinstance(v, (list, tuple, set)):
        return [str(x) for x in v]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                x = json.loads(s)
                if isinstance(x, list):
                    return [str(i) for i in x]
            except Exception:
                pass
        return [i.strip() for i in s.split(",") if i.strip()]
    return [str(v)]


def seniority_to_years(s):
    s = str(s or "").lower()
    if "intern" in s:
        return 0
    if "junior" in s:
        return 1
    if "mid" in s:
        return 3
    if "senior" in s:
        return 6
    if "lead" in s or "principal" in s:
        return 8
    return 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/train_pairs.jsonl")
    parser.add_argument("--neg-ratio", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo = "michaelozon/candidate-matching-synthetic"
    resumes = pd.read_parquet(f"hf://datasets/{repo}/resumes/train-00000-of-00001.parquet")
    jobs = pd.read_parquet(f"hf://datasets/{repo}/jobs/train-00000-of-00001.parquet")
    matches = pd.read_parquet(f"hf://datasets/{repo}/matches/train-00000-of-00001.parquet")

    resumes_map = {str(r["resume_id"]): r for _, r in resumes.iterrows()}
    jobs_map = {str(j["job_id"]): j for _, j in jobs.iterrows()}
    all_resume_ids = list(resumes_map.keys())

    rng = random.Random(args.seed)
    rows = []

    for _, m in matches.iterrows():
        job_id = str(m["job_id"])
        job = jobs_map.get(job_id)
        if job is None:
            continue

        positives = {rid for rid in to_list(m["relevant_resume_ids"]) if rid in resumes_map}
        if not positives:
            continue

        neg_pool = [rid for rid in all_resume_ids if rid not in positives]
        n_neg = min(len(neg_pool), max(1, len(positives) * args.neg_ratio))
        negatives = set(rng.sample(neg_pool, n_neg))

        required_skills = to_list(job.get("must_have_skills")) + to_list(job.get("nice_to_have_skills"))

        for rid in positives | negatives:
            r = resumes_map[rid]
            label = 1 if rid in positives else 0

            rows.append(
                {
                    "query_id": job_id,
                    "label": label,
                    "employee": {
                        "id": rid,
                        "full_name": f"resume_{rid}",
                        "position": str(r.get("role", "")),
                        "departement": str(r.get("industry", "")),
                        "skills": to_list(r.get("skills")),
                        "hire_date": None,
                        "performance_score": None,
                        "summary": str(r.get("summary", "")),
                        "experience_years": float(r.get("years_experience", 0) or 0),
                    },
                    "job": {
                        "id": job_id,
                        "title": str(job.get("job_title", "")),
                        "description": str(job.get("description", "")),
                        "required_skills": required_skills,
                        "min_experience": seniority_to_years(job.get("seniority", "")),
                    },
                }
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} pairs to {out}")


if __name__ == "__main__":
    main()
