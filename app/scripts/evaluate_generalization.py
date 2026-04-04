from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from app.ai.matcher import CandidateMatcher
from app.ai.preprocessing import preprocess_training_pairs
from app.scripts.train_matcher import load_pairs

def _parse_scenarios(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid --scenario format: {raw} (expected name=path)")
        name, path = raw.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --scenario format: {raw} (expected name=path)")
        out[name] = Path(path)
    return out

def _scenario_metrics(
    matcher: CandidateMatcher,
    pairs: list[dict[str, Any]],
    *,
    k: int,
    threshold: float,
) -> dict[str, Any]:
    clean = preprocess_training_pairs(pairs)
    x, y = matcher.feature_engineer.vectorize_pairs(clean)
    if y is None or len(y) == 0:
        raise ValueError("Scenario has no labels.")

    if not matcher.is_fitted:
        raise ValueError("Loaded model is not fitted.")

    y_prob = matcher.model.predict_proba(x)[:, 1]

    roc_auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else None
    avg_pr = average_precision_score(y, y_prob) if len(np.unique(y)) > 1 else None
    precision, recall, f1 = matcher._classification_metrics(y, y_prob, threshold=threshold)

    query_ids = [row.get("query_id") or row.get("job_id") or idx for idx, row in enumerate(pairs)]
    ranking = matcher._ranking_metrics(y, y_prob, query_ids=query_ids, k=k)

    return {
        "rows": int(len(y)),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "average_precision": float(avg_pr) if avg_pr is not None else None,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "precision_at_k": ranking.get("precision_at_k"),
        "recall_at_k": ranking.get("recall_at_k"),
        "map_at_k": ranking.get("map_at_k"),
        "ndcg_at_k": ranking.get("ndcg_at_k"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained matcher across multiple holdout scenarios.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--scenario", action="append", required=True, help="name=path_to_jsonl_or_json")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-semantic", action="store_true", help="Disable semantic embedding features during evaluation.")
    parser.add_argument("--out", default="artifacts/generalization_report.json")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    matcher = CandidateMatcher.load(model_path)
    if args.no_semantic and getattr(matcher, "feature_engineer", None) is not None:
        matcher.feature_engineer.use_semantic = False
    scenario_map = _parse_scenarios(args.scenario)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "k": int(args.k),
        "threshold": float(args.threshold),
        "semantic_enabled": bool(getattr(getattr(matcher, "feature_engineer", None), "use_semantic", False)),
        "scenarios": {},
    }
    for name, path in scenario_map.items():
        pairs = load_pairs(path)
        if not pairs:
            raise ValueError(f"Scenario '{name}' is empty: {path}")
        payload["scenarios"][name] = _scenario_metrics(
            matcher,
            pairs,
            k=int(args.k),
            threshold=float(args.threshold),
        )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
if __name__ == "__main__":
    main()
