from __future__ import annotations
import argparse
import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from app.ai.matcher import CandidateMatcher
from app.scripts.train_matcher import load_pairs

def _load_registry(path: Path) -> list[dict]:
    if not path.exists():
        return[]
    try: 
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else[]
    except Exception:
        return []
    

def main():
    parser = argparse.ArgumentParser(description="Retrain matcher with versioned artifacts")
    parser.add_argument("--input", required=True, help="Training pairs json/jsonl path.")
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="directory where model artifacts are saved"
    )
    parser.add_argument(
        "--version",
        default=None,
        help="optional model version tag(default: UTC timestamp).",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--no-semantic", action="store_true")
    parser.add_argument("--promote-latest", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    trained_at = datetime.now(timezone.utc)
    version = args.version or trained_at.strftime("%Y%m%d_%H%M%S")
    #artifacts directory and create
    artifacts_dir= Path(args.artifacts_dir).expanduser()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    versioned_model = artifacts_dir / f"matcher_{version}.joblib"
    versioned_metrics = artifacts_dir / f"matcher_metrics_{version}.json"
    stable_model = artifacts_dir / "matcher.joblib"
    stable_metrics = artifacts_dir / "matcher_metrics.json"
    registry_path = artifacts_dir / "model_registry.json"

    pairs = load_pairs(Path(args.input))
    if not pairs:
        raise ValueError("No training pairs found.")

    query_ids = [row.get("query_id") or row.get("job_id") or idx for idx, row in enumerate(pairs)]

    matcher = CandidateMatcher(use_semantic=not args.no_semantic)
    result = matcher.train(
        pairs,
        query_ids=query_ids,
        k=args.k,
        valid_size=args.valid_size,
    )
    matcher.save(versioned_model)

    metrics = asdict(result)
    metrics.update(
        {
            "model_version": version,
            "trained_at_utc": trained_at.isoformat(),
            "input_path": str(Path(args.input)),
            "artifact_path": str(versioned_model),
        }
    )
    versioned_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.promote_latest:
        shutil.copy2(versioned_model, stable_model)
        shutil.copy2(versioned_metrics, stable_metrics)

    registry = _load_registry(registry_path)
    registry.append(
        {
            "model_version": version,
            "trained_at_utc": trained_at.isoformat(),
            "artifact_path": str(versioned_model),
            "metrics_path": str(versioned_metrics),
            "promoted_to_latest": bool(args.promote_latest),
        }
    )
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Versioned model: {versioned_model}")
    print(f"Versioned metrics: {versioned_metrics}")
    print(f"Registry: {registry_path}")
    if args.promote_latest:
        print(f"Promoted latest model to: {stable_model}")

if __name__ == "__main__" :
    main()