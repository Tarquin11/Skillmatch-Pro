# SkillMatch Pro — Backend

Service FastAPI alimentant SkillMatch Pro, une plateforme de mobilité interne qui :

- Gère les employés, compétences, départements, postes et candidats (CRUD)
- Ingère les CV (PDF/DOCX), extrait les entités via un modèle NER fine-tuné, et persiste les profils structurés
- Classe les candidats par rapport aux postes via un matcher ML entraîné (avec fallback heuristique)
- Exécute une boucle d'apprentissage actif (file HITL) pour les extractions à faible confiance
- Embarque une suite MLOps : monitoring de drift, calibration, déploiement canary, évaluation holdout planifiée, gates qualité (équité, robustesse)
- Authentifie les utilisateurs avec JWT (access + refresh) et applique un RBAC sur tous les endpoints

## Stack technique

| Couche | Stack |
|---|---|
| API | Python 3.11+, FastAPI, Pydantic v2 |
| Persistance | SQLAlchemy ORM, migrations Alembic, PostgreSQL/MySQL |
| Auth | JWT (access + refresh), bcrypt, matrice RBAC |
| ML — matcher | scikit-learn / LightGBM, artefacts joblib, embeddings sentence-transformers |
| ML — NER | Hugging Face `transformers`, **XLM-RoBERTa** fine-tuné (multilingue) |
| Données synthétiques | Ollama (LLM local) pour l'annotation des certifications/projets |
| Tests | pytest (~200 tests : unitaires, e2e, équité, robustesse, adversarial, matrice RBAC) |

## Modules API principaux

| Fichier | Responsabilité |
|---|---|
| `app/api/auth.py` | Connexion, refresh, logout, verrouillage de compte |
| `app/api/employees.py` | CRUD employés + filtres |
| `app/api/department.py` | CRUD départements |
| `app/api/jobs.py` | CRUD postes + compétences requises |
| `app/api/skills.py` | CRUD du catalogue de compétences |
| `app/api/candidates.py` | Upload CV + réponse d'extraction NLP |
| `app/api/match.py` | `POST /match/job` — classement ML avec fallback heuristique |
| `app/api/learning.py` | File HITL (lister / approuver / rejeter / promouvoir) |
| `app/api/ai.py` | `/ai/model-info` — métadonnées runtime du modèle + info canary |

## Pipeline NLP des CV

La couche d'extraction des CV est **ML d'abord**, avec un fallback à base de règles pour les types d'entités qui manquent encore de données d'entraînement.

### Extraction des compétences (pilotée par le modèle)

1. **Extraction de texte** depuis PDF (pdfplumber) / DOCX (python-docx).
2. **Détection des sections** : skills, expérience, certifications, projets, langues.
3. **Inférence NER** via un XLM-RoBERTa fine-tuné (`app/services/cv_ner_inference.py`).
4. **Filtre de confiance Option-B** :
   - score ≥ 0.70 → conservé
   - 0.55 ≤ score < 0.70 → conservé uniquement si le terme est présent dans le lexique ESCO
   - score < 0.55 → rejeté
5. Le **catalogue interne** (compétences validées) est interrogé en parallèle, et les résultats fusionnés avec ceux du NER.

### Titre / certifications / projets (encore à base de règles)

Des extracteurs à base de règles couvrent le titre, les certifications et les projets en attendant un volume de données suffisant pour étendre le NER à ces classes.

### Apprentissage actif (HITL)

- Les prédictions à faible confiance sont écrites dans la table `unknown_entities` avec `status="pending"`.
- Un administrateur les consulte dans la **file de revue**, peut les approuver, les rejeter, ou les promouvoir au catalogue canonique.
- Une compétence promue rejoint le catalogue et améliore les extractions ultérieures automatiquement.

## Pipelines de données et d'entraînement

L'intégralité du pipeline d'entraînement est reproductible via les scripts de `app/scripts/` :

| Script | Rôle |
|---|---|
| `build_esco_lexicon.py` | Construit un lexique multilingue (FR + EN, ~166 000 formes de surface) à partir des CSV de la classification ESCO |
| `generate_cv_synth_ollama.py` | Génère des CV synthétiques annotés certifications/projets avec un modèle Ollama local |
| `process_ollama_certproject.py` | Convertit les sorties Ollama en JSONL labellisé pour l'entraînement |
| `auto_label_certs_projects.py` | Auto-labellisation heuristique des entités cert/projet sur les CV existants |
| `build_bio_dataset.py` | Convertit les CV labellisés en données BIO-tagging |
| `train_cv_ner.py` | Fine-tune XLM-RoBERTa sur le dataset BIO (compatible Kaggle) |
| `train_matcher.py` | Entraîne le matcher candidat-poste sur les paires de ranking |
| `retrain_matcher.py` | Retraining du matcher avec les gates de promotion |
| `evaluate_cv_extraction.py` | Évalue la qualité du parser sur un holdout labellisé |
| `evaluate_cv_quality_gates.py` | Valide les gates fixes de précision/rappel/F1/ECE |
| `run_scheduled_holdout_eval.py` | Évaluation holdout périodique (drift) |

## Démarrage rapide (local)

### 1. Environnement virtuel

```bash
python -m venv venv
source venv/bin/activate     # Linux/macOS
# .\venv\Scripts\Activate.ps1  # Windows PowerShell
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Configurer l'environnement

```bash
cp .env.example .env
```

Variables requises :
- `DATABASE_URL`
- `SECRET_KEY`

Variables AI runtime (optionnelles) :
- `AI_MODEL_AUTOLOAD=true|false`
- `AI_MODEL_PATH=artifacts/matcher.joblib`
- `AI_CANARY_ENABLED=true|false`
- `AI_CANARY_MODEL_PATH=artifacts/matcher_canary.joblib`
- `AI_CANARY_TRAFFIC_PERCENT=10`
- `ENABLE_AI_MATCHING=true|false`
- `ACTIVE_LEARNING_ENABLED=true|false`
- `ACTIVE_LEARNING_REVIEW_THRESHOLD=0.75`

### 4. Initialiser la base

```bash
python -m app.init_db
# ou avec Alembic :
alembic upgrade head
```

### 5. Lancer l'API

```bash
uvicorn app.main:app --ssl-keyfile=./localhost-key.pem --ssl-certfile=./localhost.pem
```

- Health : `https://127.0.0.1:8000/`
- Documentation OpenAPI : `https://127.0.0.1:8000/docs`

## Commandes de test

Suite complète :
```bash
pytest app/tests -q
```

Ciblé (robustesse parser + entrées adversariales) :
```bash
pytest app/tests/ai/test_cv_parser_robustness.py app/tests/ai/test_cv_parser_fuzz.py app/tests/ai/test_cv_parser_adversarial.py app/tests/test_candidates_upload_noisy.py -q
```

## Flux d'entraînement et de promotion ML

### 1. Réentraîner le matcher

```bash
python -m app.services.retrain_matcher \
    --input data/features/pairs.jsonl \
    --artifacts-dir artifacts \
    --dataset-version ds_AAAA_MM_JJ \
    --version AAAAMMJJ_01
```

Sorties :
- `artifacts/matcher_<version>.joblib`
- `artifacts/matcher_metrics_<version>.json`
- `artifacts/model_registry.json`

Si les gates qualité passent et que la promotion est activée :
- `artifacts/matcher.joblib`
- `artifacts/matcher_metrics.json`

### 2. Entraîner / fine-tuner le modèle NER

Le modèle NER s'entraîne sur Kaggle pour profiter d'un GPU gratuit :

```bash
# Localement (vérification rapide sur CPU) :
python -m app.scripts.train_cv_ner --mini --epochs 1

# Sur Kaggle : uploader bio_dataset.jsonl comme dataset,
# puis exécuter train_cv_ner.py via un notebook.
```

Les artefacts sont produits dans `data/models/cv_ner/cv_ner_final/` (model.safetensors, tokenizer, config, metrics.json).

### 3. Évaluation holdout planifiée

```bash
python -m app.scripts.run_scheduled_holdout_eval
```

Génère des rapports datés sous `artifacts/evaluations/`.

### 4. Vérification finale de release-readiness

```bash
python -m app.scripts.check_release_ready \
    --pytest-target app/tests \
    --policy app/config/promotion_policy.json \
    --out artifacts/release_readiness.json
```

La release est considérée prête uniquement si : les tests passent, le gate du dernier modèle est vert, et le dernier gate de généralisation planifié est vert et frais.

## Docker

```bash
docker build -t skillmatch-pro-back .
docker run --rm -p 8000:8000 --env-file .env skillmatch-pro-back
```

## Notes du projet

- Les routes non versionnées et les routes namespacées `/api/v1/*` sont toutes les deux activées.
- `README.md` est la version anglaise du document.
- `AI_retraining.md` contient le runbook opérationnel de retraining.
- Les artefacts volumineux (poids du modèle, données ESCO, jeux d'entraînement générés) sont gitignorés — ils sont reproductibles via les scripts ci-dessus.
