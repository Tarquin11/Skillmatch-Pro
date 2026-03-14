# Offline Data Pipeline

This pipeline is separate from the application runtime. It prepares large datasets for training:

Raw -> Clean -> Features -> Train/Val/Test splits

## Run
```bash
python pipeline/run_pipeline.py --input data/raw/source.jsonl
```

## Outputs
- `data/raw/pairs.jsonl`
- `data/clean/pairs.jsonl`
- `data/features/pairs.jsonl`
- `data/splits/train.jsonl`
- `data/splits/val.jsonl`
- `data/splits/test.jsonl`

## Notes
- For very large datasets, prefer JSONL input.
- Splits are deterministic by default using `query_id` hashing.
