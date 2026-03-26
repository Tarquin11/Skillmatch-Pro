from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _normalize_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score > 1.0:
        score = score / 100.0
    return float(min(1.0, max(0.0, score)))


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reference_predictions.csv from recent real traffic monitoring rows."
    )
    parser.add_argument("--source", default="artifacts/monitoring/current_predictions.csv")
    parser.add_argument("--out", default="artifacts/monitoring/reference_predictions.csv")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--min-rows", type=int, default=1000)
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument("--job-title", default=None, help="Optional exact job_title filter.")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    if not source.exists():
        raise FileNotFoundError(f"Monitoring source file not found: {source}")

    df = pd.read_csv(source)
    if df.empty:
        raise ValueError(f"Monitoring source file is empty: {source}")

    score_col = _pick_column(df, ["predicted_score", "predicted_fit_score", "score"])
    if score_col is None:
        raise ValueError("Source file is missing score column (expected predicted_score).")

    ts_col = _pick_column(df, ["scored_at_utc", "timestamp_utc", "timestamp"])
    if ts_col is not None:
        df["_ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    else:
        df["_ts"] = pd.NaT

    if args.job_title:
        target = args.job_title.strip().lower()
        if "job_title" in df.columns:
            df = df[df["job_title"].fillna("").astype(str).str.strip().str.lower() == target]

    if args.days > 0 and ts_col is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=int(args.days))
        df = df[df["_ts"] >= cutoff]

    df["predicted_score"] = df[score_col].map(_normalize_score)
    df = df.dropna(subset=["predicted_score"]).copy()
    if df.empty:
        raise ValueError("No valid rows after filtering/normalization.")

    if "actual_label" not in df.columns and "label" in df.columns:
        df["actual_label"] = df["label"]
    if "actual_label" not in df.columns:
        df["actual_label"] = None
    if "scoring_source" not in df.columns:
        df["scoring_source"] = "unknown"

    if args.max_rows > 0 and len(df) > int(args.max_rows):
        if ts_col is not None:
            df = df.sort_values("_ts")
        df = df.tail(int(args.max_rows))

    selected = df[["predicted_score", "actual_label", "scoring_source"]].copy()
    if len(selected) < int(args.min_rows):
        raise ValueError(
            f"Not enough rows for baseline: {len(selected)} < min_rows={args.min_rows}. "
            "Run traffic longer or lower --min-rows."
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out, index=False)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": str(source),
        "output": str(out),
        "rows_written": int(len(selected)),
        "days_window": int(args.days),
        "job_title_filter": args.job_title,
        "score_min": float(selected["predicted_score"].min()),
        "score_max": float(selected["predicted_score"].max()),
        "score_mean": float(selected["predicted_score"].mean()),
        "score_stddev": float(selected["predicted_score"].std(ddof=0)),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
