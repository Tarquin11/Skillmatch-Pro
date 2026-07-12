# SkillMatch Pro — Backend

FastAPI service powering SkillMatch Pro, an internal-mobility platform that:

- Manages employees, skills, departments, job posts, and candidates (CRUD)
- Ingests CVs (PDF/DOCX), extracts entities via a fine-tuned NER model, and stores structured profiles
- Ranks candidates against job postings via a trained ML matcher (with heuristic fallback)
- Runs an active-learning loop (HITL queue) for low-confidence extractions
- Ships an MLOps suite: drift monitoring, calibration, canary deployment, scheduled holdout evaluation, fairness/robustness gates
- Authenticates users with JWT (access + refresh) and enforces RBAC across all endpoints

## Tech stack

| Layer | Stack |
|---|---|
| API | Python 3.11+, FastAPI, Pydantic v2 |
| Persistence | SQLAlchemy ORM, Alembic migrations, PostgreSQL/MySQL |
| Auth | JWT (access + refresh tokens), bcrypt, RBAC matrix |
| ML — matcher | scikit-learn / LightGBM, joblib artifacts, sentence-transformers embeddings |
| ML — NER | Hugging Face `transformers`, fine-tuned **XLM-RoBERTa** (multilingual) |
| Synthetic data | Ollama (local LLM) for cert/project annotation |
| Tests | pytest (~200 tests covering unit, e2e, fairness, robustness, adversarial, RBAC matrix) |

## Main API modules

| File | Responsibility |
|---|---|
| `app/api/auth.py` | Login, refresh, logout, account lockout |
| `app/api/employees.py` | Employee CRUD + filters |
| `app/api/department.py` | Department CRUD |
| `app/api/jobs.py` | Job posts CRUD + required skills |
| `app/api/skills.py` | Skill catalog CRUD |
| `app/api/candidates.py` | CV upload + NLP extraction response |
| `app/api/match.py` | `POST /match/job` — ML ranking with heuristic fallback |
| `app/api/learning.py` | HITL review queue (list / approve / reject / promote) |
| `app/api/ai.py` | `/ai/model-info` — runtime model metadata + canary info |

## CV NLP pipeline

The CV-extraction layer is **ML-first**, with a rule-based fallback for entity types that lack training data.

### Skill extraction (ML-driven)

1. **Text extraction** from PDF (pdfplumber) / DOCX (python-docx).
2. **Section detection** identifies skill, experience, certification, project, and language sections.
3. **NER inference** runs a fine-tuned XLM-RoBERTa over the text (`app/services/cv_ner_inference.py`).
4. **Option-B confidence filter**:
   - Score ≥ 0.70 → kept
   - 0.55 ≤ score < 0.70 → kept only if the term is present in the ESCO lexicon
   - Score < 0.55 → dropped
5. **Catalog match** against the user's curated skills DB runs in parallel and is merged with the NER output.

### Title / certifications / projects (rule-based for now)

Rule-based extractors handle title, certifications, and projects until enough training data is available to extend the NER model to those classes.

### Active learning (HITL)

- Low-confidence NER predictions and rule-based extractions below threshold are written to the `unknown_entities` table with `status="pending"`.
- An admin reviews them in the **review queue**, can approve, reject, or promote a term to the canonical skill catalog.
- Promoted skills become part of the catalog and improve subsequent extractions automatically.

## Data & training pipelines

The full training pipeline is reproducible end-to-end via the scripts in `app/scripts/`:

| Script | Purpose |
|---|---|
| `build_esco_lexicon.py` | Build a multilingual lexicon (FR + EN, ~166 K surface forms) from the ESCO classification CSVs |
| `generate_cv_synth_ollama.py` | Generate synthetic CVs with cert/project annotations using a local Ollama model |
| `process_ollama_certproject.py` | Convert Ollama outputs into labeled JSONL for training |
| `auto_label_certs_projects.py` | Heuristic auto-labeling of cert/project entities in existing CVs |
| `build_bio_dataset.py` | Convert labeled CVs into BIO-tagged training data |
| `train_cv_ner.py` | Fine-tune XLM-RoBERTa on the BIO dataset (Kaggle-friendly) |
| `train_matcher.py` | Train the candidate–job matcher on ranking pairs |
| `retrain_matcher.py` | Retrain the matcher with promotion gates |
| `evaluate_cv_extraction.py` | Evaluate parser quality on a labeled holdout |
| `evaluate_cv_quality_gates.py` | Validate fixed precision/recall/F1/ECE gates |
| `run_scheduled_holdout_eval.py` | Periodic holdout evaluation for drift |

## Quick start (local)

### 1. Virtual environment

```bash
python -m venv venv
source venv/bin/activate     # Linux/macOS
# .\venv\Scripts\Activate.ps1  # Windows PowerShell
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Required:
- `DATABASE_URL`
- `SECRET_KEY`

Optional AI runtime variables:
- `AI_MODEL_AUTOLOAD=true|false`
- `AI_MODEL_PATH=artifacts/matcher.joblib`
- `AI_CANARY_ENABLED=true|false`
- `AI_CANARY_MODEL_PATH=artifacts/matcher_canary.joblib`
- `AI_CANARY_TRAFFIC_PERCENT=10`
- `ENABLE_AI_MATCHING=true|false`
- `ACTIVE_LEARNING_ENABLED=true|false`
- `ACTIVE_LEARNING_REVIEW_THRESHOLD=0.75`

### 4. Initialize DB

```bash
python -m app.init_db
# or with Alembic:
alembic upgrade head
```

### 5. Run API

```bash
uvicorn app.main:app --ssl-keyfile=./localhost-key.pem --ssl-certfile=./localhost.pem
```

- Health: `https://127.0.0.1:8000/`
- OpenAPI docs: `https://127.0.0.1:8000/docs`

## Test commands

Full suite:
```bash
pytest app/tests -q
```

Targeted (parser robustness + adversarial inputs):
```bash
pytest app/tests/ai/test_cv_parser_robustness.py app/tests/ai/test_cv_parser_fuzz.py app/tests/ai/test_cv_parser_adversarial.py app/tests/test_candidates_upload_noisy.py -q
```

## ML training & promotion flow

### 1. Retrain the matcher

```bash
python -m app.services.retrain_matcher \
    --input data/features/pairs.jsonl \
    --artifacts-dir artifacts \
    --dataset-version ds_YYYY_MM_DD \
    --version YYYYMMDD_01
```

Outputs:
- `artifacts/matcher_<version>.joblib`
- `artifacts/matcher_metrics_<version>.json`
- `artifacts/model_registry.json`

If quality gates pass and promotion is enabled:
- `artifacts/matcher.joblib`
- `artifacts/matcher_metrics.json`

### 2. Train / fine-tune the CV NER model

The NER model is trained on Kaggle for free GPU access:

```bash
# Locally (sanity check on CPU):
python -m app.scripts.train_cv_ner --mini --epochs 1

# On Kaggle: upload bio_dataset.jsonl as a dataset,
# then run train_cv_ner.py via a notebook.
```

Outputs land in `data/models/cv_ner/cv_ner_final/` (model.safetensors, tokenizer, config, metrics.json).

### 3. Scheduled holdout evaluation

```bash
python -m app.scripts.run_scheduled_holdout_eval
```

Generates dated reports under `artifacts/evaluations/`.

### 4. Final release-readiness check

```bash
python -m app.scripts.check_release_ready \
    --pytest-target app/tests \
    --policy app/config/promotion_policy.json \
    --out artifacts/release_readiness.json
```

Release is ready only if: tests pass, the latest model gate is green, and the latest scheduled generalization gate is green and fresh.

## Docker

```bash
docker build -t skillmatch-pro-back .
docker run --rm -p 8000:8000 --env-file .env skillmatch-pro-back
```

## Project notes

- Both unversioned routes and `/api/v1/*` namespaced routes are enabled.
- `README.fr.md` is the French version for academic / supervisor reporting.
- `AI_retraining.md` contains the operational retraining runbook.
- Large artifacts (trained model weights, ESCO data, generated training sets) are gitignored — they're regenerable via the scripts above.

The journey ends here , thank you !