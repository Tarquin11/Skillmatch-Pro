# SkillMatch Pro Backend

FastAPI backend for SkillMatch Pro.

## Prerequisites

- Python 3.10+
- PostgreSQL (or another database URL supported by SQLAlchemy)
- 'venv' created in project root

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create environment file from template:

```bash
Copy-Item .env.example .env
```

4. Edit `.env` and set real values for:
- `DATABASE_URL`
- `SECRET_KEY`

## Run Locally

```bash
uvicorn app.main:app --reload
```

API health check:

```bash
curl http://127.0.0.1:8000/
```

## Initialize Database

```bash
python -m app.init_db
```

## Run Tests

Install dev dependencies:

```bash
pip install -r requirements-dev.txt
```

Run pytest:

```bash
pytest -q
```

## Docker (Simple)

Build:

```bash
docker build -t skillmatch-pro-back .
```

Run:

```bash
docker run --rm -p 8000:8000 --env-file .env skillmatch-pro-back
```

## Database Migrations (Alembic)

Set env vars first (`DATABASE_URL`, `SECRET_KEY`), then run:

```bash
alembic upgrade head

## Run Locally
```markdown

Model/version metadata endpoint:
- `GET /ai/model-info` (auth required)

Retraining process:
- See `docs/ai_retraining.md`

Run retraining:
```bash
python -m app.scripts.retrain_matcher --input artifacts/training_pairs.jsonl --artifacts-dir artifacts
