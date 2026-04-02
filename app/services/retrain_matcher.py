from __future__ import annotations
import argparse
from html import parser
from html import parser
import json
import shutil
from typing import Any
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
from app.ai.matcher import CandidateMatcher
from app.scripts.train_matcher import load_pairs
from app.services.promotion_gates import build_promotion_gate_report, load_gate_policy

def _load_registry(path: Path) -> list[dict]:
    if not path.exists():
        return[]
    try: 
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else[]
    except Exception:
        return []
    
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()
    
def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except(TypeError, ValueError):
        return None
    
def _evaluate_metric_gates(
    metrics: dict[str, Any],
    *,
    min_roc_auc: float,
    min_f1: float,
    min_map_at_k: float,
) -> tuple[bool, list[str], dict[str, Any]]:
    thresholds = {
        "roc_auc": float(min_roc_auc),
        "f1": float(min_f1),
        "map_at_k": float(min_map_at_k),
    }
    results: dict[str, Any] = {}
    failures: list[str] = []
    for metric_name, min_required in thresholds.items():
        value = _safe_float(metrics.get(metric_name))
        passed = value is not None and value >= min_required
        results[metric_name] = {
            "value": value,
            "min_required": min_required,
            "passed": passed,
        }
        if not passed:
            shown = "None" if value is None else f"{value:.6f}"
            failures.append(
                f"metric_gate_failed {metric_name}={shown} < min_required={min_required:.6f}"
            )
    return len(failures) == 0, failures, {
        "thresholds": thresholds,
        "results": results,
    }

def _evaluate_drift_gates(
    drift_report_path: Path,
    *,
    max_pos_ratio_delta: float,
    max_missing_employee_skills_delta: float,
    max_missing_job_skills_delta: float,
) -> tuple[bool, list[str], dict[str, Any]]:
    thresholds = {
        "pos_ratio_delta": float(max_pos_ratio_delta),
        "missing_employee_skills_delta": float(max_missing_employee_skills_delta),
        "missing_job_skills_delta": float(max_missing_job_skills_delta),
    }
    detail: dict[str, Any] = {
        "enabled": True,
        "report_path": str(drift_report_path),
        "thresholds": thresholds,
        "results": {},
        "passed": False,
    }
    failures: list[str] = []

    if not drift_report_path.exists():
        failures.append(f"drift_gate_failed report_not_found path={drift_report_path}")
        return False, failures, detail
    try:
        payload = json.loads(drift_report_path.read_text(encoding="utf-8"))
    except Exception:
        failures.append(f"drift_gate_failed report_unreadable path={drift_report_path}")
        return False, failures, detail
    if not isinstance(payload, dict):
        failures.append(f"drift_gate_failed report_invalid_json path={drift_report_path}")
        return False, failures, detail
    results: dict[str, Any] = {}
    for field, max_abs in thresholds.items():
        delta = _safe_float(payload.get(field))
        abs_delta = abs(delta) if delta is not None else None
        passed = abs_delta is not None and abs_delta <= max_abs
        results[field] = {
            "delta": delta,
            "abs_delta": abs_delta,
            "max_allowed_abs": max_abs,
            "passed": passed,
        }
        if not passed:
            shown = "None" if delta is None else f"{delta:.6f}"
            failures.append(
                f"drift_gate_failed {field}={shown} (abs max allowed={max_abs:.6f})"
            )
    detail["results"] = results
    detail["passed"] = len(failures) == 0
    return detail["passed"], failures, detail

def _build_promotion_gate_report(
    *,
    metrics: dict[str, Any],
    drift_report_path: Path,
    min_roc_auc: float,
    min_f1: float,
    min_map_at_k: float,
    max_pos_ratio_delta: float,
    max_missing_employee_skills_delta: float,
    max_missing_job_skills_delta: float,
    skip_drift_gates: bool,
) -> dict[str, Any]:
    metric_passed, metric_failures, metric_detail = _evaluate_metric_gates(
        metrics,
        min_roc_auc=min_roc_auc,
        min_f1=min_f1,
        min_map_at_k=min_map_at_k,
    )
    if skip_drift_gates:
        drift_passed = True
        drift_failures: list[str] = []
        drift_detail: dict[str, Any] = {
            "enabled": False,
            "report_path": str(drift_report_path),
            "thresholds": {
                "pos_ratio_delta": float(max_pos_ratio_delta),
                "missing_employee_skills_delta": float(max_missing_employee_skills_delta),
                "missing_job_skills_delta": float(max_missing_job_skills_delta),
            },
            "results": {},
            "passed": True,
            "skipped": True,
        }
    else:
        drift_passed, drift_failures, drift_detail = _evaluate_drift_gates(
            drift_report_path,
            max_pos_ratio_delta=max_pos_ratio_delta,
            max_missing_employee_skills_delta=max_missing_employee_skills_delta,
            max_missing_job_skills_delta=max_missing_job_skills_delta,
        )
        drift_detail["skipped"] = False

    failures = [*metric_failures, *drift_failures]
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": metric_passed and drift_passed,
        "failures": failures,
        "metric_gates": {
            "passed": metric_passed,
            **metric_detail,
        },
        "drift_gates": drift_detail,
    }
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
    parser.add_argument(
        "--dataset-version",
        default=None,
        help="optional dataset version string to store in metrics.",
    )
    parser.add_argument("--gates-config", default="app/config/promotion_policy.json")
    parser.add_argument("--drift-report", default=None, help="Override drift report path from gate config.")
    parser.add_argument("--generalization-report", default=None, help="Override generalization report path from gate config.")
    parser.add_argument("--robustness-report", default=None, help="Override robustness report path from gate config.")
    parser.add_argument("--skip-drift-gates", action="store_true")
    parser.add_argument("--skip-generalization-gates", action="store_true")
    parser.add_argument("--skip-robustness-gates", action="store_true")
    parser.add_argument("--min-roc-auc", type=float, default=None)
    parser.add_argument("--min-f1", type=float, default=None)
    parser.add_argument("--min-map-at-k", type=float, default=None)
    parser.add_argument("--max-pos-ratio-delta", type=float, default=None)
    parser.add_argument("--max-missing-employee-skills-delta", type=float, default=None)
    parser.add_argument("--max-missing-job-skills-delta", type=float, default=None)
    parser.add_argument("--max-crash-rate", type=float, default=None)
    parser.add_argument("--min-schema-valid-rate", type=float, default=None)
    parser.add_argument("--fail-on-gate-fail", action=argparse.BooleanOptionalAction, default=False)
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
    metrics["dataset_version"] = args.dataset_version or "didnt put yet ! "
    metrics["dataset_sha256"] = _sha256(Path(args.input))

    gate_overrides: dict[str, Any] = {}
    if args.drift_report:
        gate_overrides.setdefault("drift", {})["report_path"] = str(args.drift_report)
    if args.generalization_report:
        gate_overrides.setdefault("generalization", {})["report_path"] = str(args.generalization_report)
    if args.robustness_report:
        gate_overrides.setdefault("robustness", {})["report_path"] = str(args.robustness_report)
    if args.skip_drift_gates:
        gate_overrides.setdefault("drift", {})["enabled"] = False
    if args.skip_generalization_gates:
        gate_overrides.setdefault("generalization", {})["enabled"] = False
    if args.skip_robustness_gates:
        gate_overrides.setdefault("robustness", {})["enabled"] = False
    if args.min_roc_auc is not None:
        gate_overrides.setdefault("metrics", {}).setdefault("minimums", {})["roc_auc"] = float(args.min_roc_auc)
    if args.min_f1 is not None:
        gate_overrides.setdefault("metrics", {}).setdefault("minimums", {})["f1"] = float(args.min_f1)
    if args.min_map_at_k is not None:
        gate_overrides.setdefault("metrics", {}).setdefault("minimums", {})["map_at_k"] = float(args.min_map_at_k)
    if args.max_pos_ratio_delta is not None:
        gate_overrides.setdefault("drift", {}).setdefault("max_abs_delta", {})["pos_ratio_delta"] = float(args.max_pos_ratio_delta)
    if args.max_missing_employee_skills_delta is not None:
        gate_overrides.setdefault("drift", {}).setdefault("max_abs_delta", {})["missing_employee_skills_delta"] = float(args.max_missing_employee_skills_delta)
    if args.max_missing_job_skills_delta is not None:
        gate_overrides.setdefault("drift", {}).setdefault("max_abs_delta", {})["missing_job_skills_delta"] = float(args.max_missing_job_skills_delta)
    if args.max_crash_rate is not None:
        gate_overrides.setdefault("robustness", {}).setdefault("maximums", {})["crash_rate"] = float(args.max_crash_rate)
    if args.min_schema_valid_rate is not None:
        gate_overrides.setdefault("robustness", {}).setdefault("minimums", {})["schema_valid_rate"] = float(args.min_schema_valid_rate)

    gate_policy = load_gate_policy(args.gates_config, overrides=gate_overrides)
    gate_report = build_promotion_gate_report(metrics=metrics, policy=gate_policy)

    metrics["promotion_gate_policy"] = gate_policy
    metrics["promotion_gate"] = gate_report

    versioned_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    should_promote = bool(args.promote_latest and gate_report["passed"])
    if should_promote:
        shutil.copy2(versioned_model, stable_model)
        shutil.copy2(versioned_metrics, stable_metrics)
    elif args.promote_latest:
        print("Promotion blocked by gates:")
        for reason in gate_report.get("failures", []):
            print(f" - {reason}")
    registry = _load_registry(registry_path)
    registry.append(
        {
            "model_version": version,
            "trained_at_utc": trained_at.isoformat(),
            "artifact_path": str(versioned_model),
            "metrics_path": str(versioned_metrics),
            "promoted_to_latest": bool(should_promote),
            "promotion_gate_passed": bool(gate_report.get("passed", False)),
            "promotion_gate_failures": list(gate_report.get("failures", [])),
        }
    )
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Versioned model: {versioned_model}")
    print(f"Versioned metrics: {versioned_metrics}")
    print(f"Registry: {registry_path}")
    if should_promote:
        print(f"Promoted latest model to: {stable_model}")
    elif args.promote_latest:
        print("latest artifact not updated (promotion gates failed)")
    if args.promote_latest and not gate_report["passed"] and args.fail_on_gate_fail:
        raise SystemExit("Promotion gets failed; latest artifact not updated ! ")

if __name__ == "__main__" :
    main()
