#drift comparison script

from __future__ import annotations
import argparse
import json
from pathlib import Path

def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare data profiles for drift")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--out", default="artifacts/drift_report.json")
    args = parser.parse_args()

    base = _load(Path(args.baseline))
    curr = _load(Path(args.current))

    report = {
        "pos_ratio_delta": round(curr["pos_ratio"] - base["pos_ratio"], 6),
        "missing_employee_skills_delta": round(
            curr["missing_employee_skills_ratio"] - base["missing_employee_skills_ratio"], 6
        ),
        "missing_job_skills_delta": round(
            curr["missing_job_skills_ratio"] - base["missing_job_skills_ratio"], 6
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()