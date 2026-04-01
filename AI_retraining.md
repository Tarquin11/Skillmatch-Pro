# AI Retraining Process

## 1. Prepare labeled data
Accepted format:
- JSON array of pairs
- JSONL (one pair per line)
- JSON object with `{"pairs": [...]}`

Each pair must include:
- `employee`
- `job`
- `label` (0/1)
Optional:
- `query_id` or `job_id`

## 2. Data Prep (Clean, Split, Validate)

Build clean/feature-ready pairs and splits:

```bash
python pipeline/run_pipeline.py --input data/raw/source.jsonl

## Incident Checklist: Model Degradation

### Scope
Use this checklist when ranking quality drops, gates fail, or production behavior degrades.

### 1) Trigger Conditions (open incident if any is true)
- [] Latest `artifacts/evaluations/generalization_gate_*.json` has `"passed": false`
- [] `promotion_gate.passed = false` in latest `artifacts/matcher_metrics_*.json`
- [] Drift report exceeds policy thresholds in `artifacts/drift_report.json`
- [] Online KPI drop (precision/CTR/conversion) exceeds agreed threshold
- [] Error rate or latency spike on matching endpoints

### 2) Immediate Containment (0-15 min)
- [] Stop model promotion to `artifacts/matcher.joblib`
- [] Set canary traffic to `0` (or disable canary) in `.env`
- [] Restart API service to apply canary rollback
- [] If impact is high, restore last known good model to `artifacts/matcher.joblib`
- [] Confirm `/ai/model-info` shows expected active model and canary settings

### 3) Evidence Collection (15-30 min)
- [] Save latest gate report (`artifacts/evaluations/generalization_gate_*.json`)
- [] Save latest generalization report (`artifacts/evaluations/generalization_report_*.json`)
- [] Save latest training metrics (`artifacts/matcher_metrics_*.json`)
- [] Save `artifacts/model_registry.json`
- [] Save recent API logs for `/match/job` and `/ai/model-info`

### 4) Triage (30-60 min)
- [] Compare current metrics vs last promoted model (ROC-AUC, F1, MAP@K, NDCG@K)
- [] Check data freshness and schema consistency
- [] Re-run leakage and data-profile checks
- [] Verify no config drift in `app/config/promotion_policy.json`
- [] Verify artifact paths and model version alignment

### 5) Recovery Decision
- [] If issue is data/config: fix data or policy, re-evaluate, then retrain
- [] If issue is model-specific: rollback to previous stable model
- [] If issue is uncertain: keep canary at 0 and maintain rollback model

### 6) Validation Before Re-Promote
- [] Re-run holdout evaluation
- [] Confirm all required scenarios pass policy thresholds
- [] Confirm drift checks pass
- [] Confirm canary monitoring fields are correct in `/ai/model-info`
- [] Re-enable canary gradually (example: 5% -> 10% -> 25% -> 50% -> 100%)

### 7) Communication Template
- [] Incident opened at: `<UTC time>`
- [] Impact summary: `<who/what affected>`
- [] Containment applied: `<canary=0 / rollback version>`
- [] Root cause status: `<investigating / identified / fixed>`
- [] Next update ETA: `<time>`

### 8) Post-Incident (within 24h)
- [x] Write RCA (root cause, detection gap, prevention)
- [] Add/adjust gate thresholds or scenario tests if needed
- [] Update runbook commands and owners
- [] Create follow-up tasks with due dates


