# SkillMatch Pro Backend

Le backend de SkillMatch Pro est un service FastAPI dédié au matching interne des talents:
- Gestion des employés et compétences (CRUD)
- Upload de CV et extraction NLP
- Classement IA pour le matching poste/candidat
- Authentification + RBAC (admin, recruiter, user)
- Gates qualité offline (métriques, drift, généralisation, robustesse)
- Évaluation holdout planifiée et vérification de readiness release

## Stack Technique

- Python + FastAPI
- SQLAlchemy + Alembic
- Pydantic v2
- Auth JWT
- Modèle de matching basé sur scikit-learn
- Automatisation de tests avec pytest

## Modules API Principaux

- `app/api/auth.py` -> signup/login/refresh/me/gestion des rôles
- `app/api/employees.py` -> CRUD employés
- `app/api/jobs.py` -> endpoints postes
- `app/api/skills.py` -> endpoints compétences
- `app/api/candidates.py` -> upload CV + réponse d’extraction
- `app/api/match.py` -> `/match/job` et `/match/jobs`
- `app/api/ai.py` -> `/ai/model-info` (métadonnées modèle + canary)

## Démarrage Rapide (Local)

### 1. Créer et activer l’environnement virtuel

PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Installer les dépendances

```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Configurer l’environnement

```powershell
Copy-Item .env.example .env
```

Configurer au minimum:
- `DATABASE_URL`
- `SECRET_KEY`

Variables IA optionnelles:
- `AI_MODEL_AUTOLOAD=true|false`
- `AI_MODEL_PATH=artifacts/matcher.joblib`
- `AI_CANARY_ENABLED=true|false`
- `AI_CANARY_MODEL_PATH=artifacts/matcher_canary.joblib`
- `AI_CANARY_TRAFFIC_PERCENT=10`

### 4. Initialiser la base

```powershell
python -m app.init_db
```

ou avec Alembic:

```powershell
alembic upgrade head
```

### 5. Lancer l’API

```powershell
uvicorn app.main:app --reload
```

Health check:

```powershell
curl http://127.0.0.1:8000/
```

Swagger:
- `http://127.0.0.1:8000/docs`

## Commandes de Tests

Suite complète:

```powershell
pytest app/tests -q
```

Suite robustesse (équivalente CI) en local:

```powershell
pytest app/tests/ai/test_cv_parser_robustness.py app/tests/ai/test_cv_parser_fuzz.py app/tests/test_candidates_upload_noisy.py -q
```

## Flux IA: Entraînement et Promotion

### 1. Réentraîner le modèle et évaluer les gates

```powershell
python -m app.services.retrain_matcher --input data/features/pairs.jsonl --artifacts-dir artifacts --dataset-version ds_YYYY_MM_DD --version YYYYMMDD_01
```

Artifacts générés:
- `artifacts/matcher_<version>.joblib`
- `artifacts/matcher_metrics_<version>.json`
- `artifacts/model_registry.json`

Si les gates passent et que la promotion est activée:
- `artifacts/matcher.joblib`
- `artifacts/matcher_metrics.json`

### 2. Évaluation holdout planifiée

Scripts:
- `app/scripts/run_scheduled_holdout_eval.py`
- helper PowerShell: `run_holdout_eval.ps1`

Artifacts générés:
- `artifacts/evaluations/generalization_report_<timestamp>.json`
- `artifacts/evaluations/generalization_gate_<timestamp>.json`
- `artifacts/generalization_report.json` (copie latest)

### 3. Gate final de release

```powershell
python -m app.scripts.check_release_ready --pytest-target app/tests --policy app/config/promotion_policy.json --out artifacts/release_readiness.json
```

La release est prête uniquement si:
- les tests passent,
- le gate modèle le plus récent est vert,
- le gate de généralisation planifié le plus récent est vert et non obsolète.

## Évaluation Extraction CV

Construire un jeu labellisé depuis Hugging Face:

```powershell
python -m app.scripts.build_cv_eval_set_from_hf
```

Évaluer la qualité d’extraction:

```powershell
python -m app.scripts.evaluate_cv_extraction --labels-jsonl data/labels/cv_extraction_hf_labels.jsonl --out artifacts/cv_extraction_report_hf.json
```

Évaluer les KPI de robustesse du parser:

```powershell
python -m app.scripts.evaluate_cv_robustness --labels-jsonl data/labels/cv_extraction_hf_labels.jsonl --out artifacts/cv_robustness_report.json
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

## Notes Projet

- Les routes legacy non versionnées et les routes `/api/v1` sont toutes les deux actives.
- `README.fr.md` est prévu pour le suivi avec l’encadrant.
- `AI_retraining.md` contient le runbook opérationnel de réentraînement.

