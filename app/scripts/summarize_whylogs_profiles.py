from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROFILE_PATTERN = re.compile(
    r"^profile_(?P<job>.+)_(?P<stamp>\d{8}T\d{6}Z)\.bin$",
    re.IGNORECASE,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_filename(path: Path) -> tuple[str, datetime]:
    match = PROFILE_PATTERN.match(path.name)
    if not match:
        ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return "unknown", ts
    raw_job = match.group("job")
    stamp = match.group("stamp")
    dt = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return raw_job.replace("_", " "), dt


def _read_profile_summary(path: Path) -> dict[str, Any] | None:
    try:
        from whylogs.core.view.dataset_profile_view import DatasetProfileView
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("whylogs is required to summarize profile binaries") from exc

    try:
        view = DatasetProfileView.read(str(path))
        col = view.get_column("predicted_score")
        if col is None:
            return None
        return col.to_summary_dict()
    except Exception:
        return None


def _profile_row(path: Path, summary: dict[str, Any], job: str, ts: datetime) -> dict[str, Any]:
    return {
        "file_name": path.name,
        "job_title": job,
        "timestamp_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "day_utc": ts.strftime("%Y-%m-%d"),
        "n": int(_safe_float(summary.get("distribution/n", summary.get("counts/n", 0)), 0.0)),
        "mean": _safe_float(summary.get("distribution/mean")),
        "stddev": _safe_float(summary.get("distribution/stddev")),
        "min": _safe_float(summary.get("distribution/min")),
        "max": _safe_float(summary.get("distribution/max")),
        "q10": _safe_float(summary.get("distribution/q_10")),
        "median": _safe_float(summary.get("distribution/median")),
        "q90": _safe_float(summary.get("distribution/q_90")),
    }


def _daily_drift_aggregate(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        day = str(event.get("day_utc", "")).strip()
        if day:
            by_day[day].append(event)

    out: dict[str, dict[str, Any]] = {}
    for day, group in by_day.items():
        psi_vals = [float(e["score_psi"]) for e in group if e.get("score_psi") is not None]
        cal_vals = [float(e["calibration_drift"]) for e in group if e.get("calibration_drift") is not None]
        out[day] = {
            "events_count": len(group),
            "score_psi_mean": round(sum(psi_vals) / len(psi_vals), 6) if psi_vals else None,
            "score_psi_max": round(max(psi_vals), 6) if psi_vals else None,
            "calibration_drift_mean": round(sum(cal_vals) / len(cal_vals), 6) if cal_vals else None,
            "calibration_drift_abs_max": round(max(abs(v) for v in cal_vals), 6) if cal_vals else None,
        }
    return out


def _daily_aggregate(
    rows: list[dict[str, Any]],
    *,
    drift_by_day: dict[str, dict[str, Any]],
    mean_warning_threshold: float,
    std_warning_threshold: float,
    retrain_psi_threshold: float,
    retrain_calibration_threshold: float,
) -> list[dict[str, Any]]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_day[row["day_utc"]].append(row)

    out: list[dict[str, Any]] = []
    for day in sorted(by_day.keys()):
        group = by_day[day]
        total_n = sum(int(r["n"]) for r in group)
        weighted_mean_num = sum(float(r["mean"]) * int(r["n"]) for r in group)
        weighted_std_num = sum(float(r["stddev"]) * int(r["n"]) for r in group)
        min_score = min(float(r["min"]) for r in group) if group else 0.0
        max_score = max(float(r["max"]) for r in group) if group else 0.0
        jobs = Counter(r["job_title"] for r in group)
        mean_weighted = round(weighted_mean_num / total_n, 6) if total_n else 0.0
        stddev_weighted = round(weighted_std_num / total_n, 6) if total_n else 0.0
        calibration_warning = (mean_weighted > mean_warning_threshold) or (stddev_weighted < std_warning_threshold)
        day_drift = drift_by_day.get(day, {})
        psi_max = day_drift.get("score_psi_max")
        calibration_abs_max = day_drift.get("calibration_drift_abs_max")
        retrain_reasons: list[str] = []
        if psi_max is not None and float(psi_max) >= retrain_psi_threshold:
            retrain_reasons.append("score_psi")
        if calibration_abs_max is not None and float(calibration_abs_max) >= retrain_calibration_threshold:
            retrain_reasons.append("calibration_drift")
        retrain_trigger = len(retrain_reasons) > 0

        out.append(
            {
                "day_utc": day,
                "profiles_count": len(group),
                "rows_total": total_n,
                "mean_weighted": mean_weighted,
                "stddev_weighted": stddev_weighted,
                "calibration_warning": calibration_warning,
                "min_score": round(min_score, 6),
                "max_score": round(max_score, 6),
                "top_jobs": jobs.most_common(10),
                "drift_events_count": int(day_drift.get("events_count", 0)),
                "score_psi_mean": day_drift.get("score_psi_mean"),
                "score_psi_max": psi_max,
                "calibration_drift_mean": day_drift.get("calibration_drift_mean"),
                "calibration_drift_abs_max": calibration_abs_max,
                "retrain_trigger": retrain_trigger,
                "retrain_trigger_reasons": retrain_reasons,
            }
        )
    return out


def _read_drift_events(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], 0

    rows: list[dict[str, Any]] = []
    failures = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except Exception:
                failures += 1
                continue

            ts = str(item.get("timestamp_utc") or item.get("generated_at_utc") or "").strip()
            day = ts[:10] if len(ts) >= 10 else ""
            if not day:
                failures += 1
                continue

            rows.append(
                {
                    "day_utc": day,
                    "score_psi": _safe_float(item.get("score_psi"), default=None),
                    "calibration_drift": _safe_float(item.get("calibration_drift"), default=None),
                }
            )
    return rows, failures


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["file_name", "job_title", "timestamp_utc", "day_utc", "n", "mean", "stddev", "min", "max", "q10", "median", "q90"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize whylogs profile binaries into daily drift artifacts.")
    parser.add_argument("--input-dir", default="artifacts/monitoring/whylogs")
    parser.add_argument("--out-json", default="artifacts/monitoring/whylogs_daily_summary.json")
    parser.add_argument("--out-csv", default="artifacts/monitoring/whylogs_profiles.csv")
    parser.add_argument("--drift-events", default="artifacts/monitoring/drift_metrics.jsonl")
    parser.add_argument("--mean-warning-threshold", type=float, default=0.95)
    parser.add_argument("--std-warning-threshold", type=float, default=0.03)
    parser.add_argument("--retrain-psi-threshold", type=float, default=0.2)
    parser.add_argument("--retrain-calibration-threshold", type=float, default=0.05)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    drift_events_path = Path(args.drift_events)

    files = sorted(input_dir.glob("*.bin"))
    rows: list[dict[str, Any]] = []
    failures = 0

    for path in files:
        job, ts = _parse_filename(path)
        summary = _read_profile_summary(path)
        if summary is None:
            failures += 1
            continue
        rows.append(_profile_row(path, summary, job, ts))

    drift_events, drift_failures = _read_drift_events(drift_events_path)
    drift_by_day = _daily_drift_aggregate(drift_events)
    daily = _daily_aggregate(
        rows,
        drift_by_day=drift_by_day,
        mean_warning_threshold=float(args.mean_warning_threshold),
        std_warning_threshold=float(args.std_warning_threshold),
        retrain_psi_threshold=float(args.retrain_psi_threshold),
        retrain_calibration_threshold=float(args.retrain_calibration_threshold),
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_dir": str(input_dir),
        "drift_events_path": str(drift_events_path),
        "files_scanned": len(files),
        "files_parsed": len(rows),
        "files_failed": failures,
        "drift_events_scanned": len(drift_events) + drift_failures,
        "drift_events_parsed": len(drift_events),
        "drift_events_failed": drift_failures,
        "thresholds": {
            "calibration_warning_mean_gt": float(args.mean_warning_threshold),
            "calibration_warning_std_lt": float(args.std_warning_threshold),
            "retrain_score_psi_gte": float(args.retrain_psi_threshold),
            "retrain_calibration_drift_abs_gte": float(args.retrain_calibration_threshold),
        },
        "days": daily,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(out_csv, rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
