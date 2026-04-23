# SkillMatch Pro Backend

SkillMatch Pro backend is a FastAPI service for internal talent matching:
- Employee and skill management (CRUD)
- CV upload and NLP extraction
- AI ranking for job/candidate matching
- Auth + RBAC (admin, recruiter, user)
- Offline quality gates (metrics, drift, generalization, robustness)
- Scheduled holdout evaluation and release readiness checks

## Tech Stack

- Python + FastAPI
- SQLAlchemy + Alembic
- Pydantic v2
- JWT auth
- scikit-learn based matcher
- pytest for test automation

## Main API Modules

- `app/api/auth.py` -> signup/login/refresh/me/role management
- `app/api/employees.py` -> employees CRUD
- `app/api/jobs.py` -> jobs endpoints
- `app/api/skills.py` -> skills endpoints
- `app/api/candidates.py` -> CV upload + extraction response
- `app/api/match.py` -> `/match/job` and `/match/jobs`
- `app/api/ai.py` -> `/ai/model-info` (model metadata + canary info)

## Quick Start (Local)

### 1. Create and activate virtual environment

PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Configure environment

```powershell
Copy-Item .env.example .env
```

Set at least:
- `DATABASE_URL`
- `SECRET_KEY`

Optional AI runtime vars:
- `AI_MODEL_AUTOLOAD=true|false`
- `AI_MODEL_PATH=artifacts/matcher.joblib`
- `AI_CANARY_ENABLED=true|false`
- `AI_CANARY_MODEL_PATH=artifacts/matcher_canary.joblib`
- `AI_CANARY_TRAFFIC_PERCENT=10`

### 4. Initialize DB

```powershell
python -m app.init_db
```

or with Alembic:

```powershell
alembic upgrade head
```

### 5. Run API

```powershell
uvicorn app.main:app --ssl-keyfile=./localhost-key.pem --ssl-certfile=./localhost.pem
```

Health check:

```powershell
curl https://127.0.0.1:8000/
```

Swagger:
- `https://127.0.0.1:8000/docs`

## Test Commands

Run full test suite:

```powershell
pytest app/tests -q
```

Run robustness CI suite locally:

```powershell
pytest app/tests/ai/test_cv_parser_robustness.py app/tests/ai/test_cv_parser_fuzz.py app/tests/test_candidates_upload_noisy.py -q
```

## AI Training and Promotion Flow

### 1. Retrain model and evaluate gates

```powershell
python -m app.services.retrain_matcher --input data/features/pairs.jsonl --artifacts-dir artifacts --dataset-version ds_YYYY_MM_DD --version YYYYMMDD_01
```

This generates:
- `artifacts/matcher_<version>.joblib`
- `artifacts/matcher_metrics_<version>.json`
- `artifacts/model_registry.json`

If gates pass and promotion is enabled, stable artifacts are updated:
- `artifacts/matcher.joblib`
- `artifacts/matcher_metrics.json`

### 2. Scheduled holdout evaluation

Use:
- `app/scripts/run_scheduled_holdout_eval.py`
- or the helper PowerShell script: `run_holdout_eval.ps1`

Generated artifacts:
- `artifacts/evaluations/generalization_report_<timestamp>.json`
- `artifacts/evaluations/generalization_gate_<timestamp>.json`
- `artifacts/generalization_report.json` (latest copy)

### 3. Final release gate

```powershell
python -m app.scripts.check_release_ready --pytest-target app/tests --policy app/config/promotion_policy.json --out artifacts/release_readiness.json
```

Release is considered ready only when:
- tests pass,
- latest model gate is green,
- latest scheduled generalization gate is green and fresh.

## CV Extraction Evaluation

Build labeled evaluation set from Hugging Face:

```powershell
python -m app.scripts.build_cv_eval_set_from_hf
```

Evaluate extraction quality:

```powershell
python -m app.scripts.evaluate_cv_extraction --labels-jsonl data/labels/cv_extraction_hf_labels.jsonl --out artifacts/cv_extraction_report_hf.json
```

Evaluate parser robustness KPIs:

```powershell
python -m app.scripts.evaluate_cv_robustness --labels-jsonl data/labels/cv_extraction_hf_labels.jsonl --out artifacts/cv_robustness_report.json
```

Run fixed validation quality gates (technical precision/recall/F1, semantic augment FP rate, boards exclusivity, ECE):

```powershell
python -m app.scripts.evaluate_cv_quality_gates --gold-jsonl artifacts/gold/cv_quality_validation.jsonl --out artifacts/reports/cv_quality_gates_report.json
```

## Docker

Build:

```powershell
docker build -t skillmatch-pro-back .
```

Run:

```powershell
docker run --rm -p 8000:8000 --env-file .env skillmatch-pro-back
```

## Project Notes

- Legacy unversioned routes and `/api/v1` namespaced routes are both enabled.
- `README.fr.md` contains the French version for supervisor/academic reporting.
- `AI_retraining.md` contains the operational retraining runbook.
