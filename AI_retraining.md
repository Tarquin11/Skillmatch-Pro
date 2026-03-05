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

## 2. Train a new version
```bash
python -m app.scripts.retrain_matcher --input artifacts/training_pairs.jsonl --artifacts-dir artifacts
