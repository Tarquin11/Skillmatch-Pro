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

