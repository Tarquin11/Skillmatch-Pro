import argparse
import json
from dataclasses import asdict
from pathlib import Path

from app.ai.matcher import CandidateMatcher


def load_pairs(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    data = json.loads(text)
    if isinstance(data, dict) and "pairs" in data:
        return data["pairs"]
    if isinstance(data, list):
        return data
    raise ValueError("Training input must be JSON array, JSONL, or {'pairs': [...]}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="artifacts/matcher.joblib")
    parser.add_argument("--metrics-out", default="artifacts/matcher_metrics.json")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--no-semantic", action="store_true")
    args = parser.parse_args()

    pairs = load_pairs(Path(args.input))
    if not pairs:
        raise ValueError("No training pairs found.")

    query_ids = [row.get("query_id") or row.get("job_id") or idx for idx, row in enumerate(pairs)]

    matcher = CandidateMatcher(use_semantic=not args.no_semantic)
    result = matcher.train(pairs, query_ids=query_ids, k=args.k, valid_size=args.valid_size)
    matcher.save(args.output)

    out = Path(args.metrics_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")

    print("Training complete")
    print(json.dumps(asdict(result), indent=2))
    print(f"Model saved to: {args.output}")
    print(f"Metrics saved to: {args.metrics_out}")


if __name__ == "__main__":
    main()
