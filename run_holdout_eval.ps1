Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location "C:\Users\zarro\OneDrive\Desktop\PFE\skillmatch-pro-back"

& ".\venv\Scripts\python.exe" -m app.scripts.run_scheduled_holdout_eval `
  --model artifacts/matcher.joblib `
  --scenario iid_val=data/splits/val.jsonl `
  --scenario candidate_disjoint_val=data/splits_by_candidate/val.jsonl `
  --scenario job_disjoint_val=data/splits_by_job/val.jsonl `
  --policy app/config/promotion_policy.json `
  --max-holdout-age-hours 36 `
  --out-dir artifacts/evaluations `
  --latest-out artifacts/generalization_report.json
