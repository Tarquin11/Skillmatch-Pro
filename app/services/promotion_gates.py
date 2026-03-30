from __future__ import annotations
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_GATE_POLICY: dict[str, Any] = {
    "metrics": {
        "enabled": True,
        "minimums": {
            "roc_auc": 0.90,
            "f1": 0.85,
            "map_at_k": 0.85,
        },
    },
    "drift": {
        "enabled": True,
        "report_path": "artifacts/drift_report.json",
        "max_abs_delta": {
            "pos_ratio_delta": 0.05,
            "missing_employee_skills_delta": 0.05,
            "missing_job_skills_delta": 0.05,
        },
    },
    "generalization": {
        "enabled": False,
        "report_path": "artifacts/generalization_report.json",
        "required_scenarios": {}
    },
}

def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out
def load_gate_policy(path: str | Path | None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = copy.deepcopy(DEFAULT_GATE_POLICY)
    if path:
        cfg_path = Path(path).expanduser()
        if cfg_path.exists():
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"Gate policy must be a JSON object: {cfg_path}")
            policy = _deep_merge(policy, data)
    if overrides:
        policy = _deep_merge(policy, overrides)
    return policy
def _evaluate_minimums(
    actual: dict[str, Any],
    minimums: dict[str, Any],
    *,
    prefix: str,
) -> tuple[bool, list[str], dict[str, Any]]:
    results: dict[str, Any] = {}
    failures: list[str] = []

    for metric_name, threshold_raw in (minimums or {}).items():
        threshold = _safe_float(threshold_raw)
        value = _safe_float(actual.get(metric_name))
        passed = threshold is not None and value is not None and value >= threshold

        results[metric_name] = {
            "value": value,
            "min_required": threshold,
            "passed": passed,
        }

        if threshold is None:
            failures.append(f"{prefix} invalid_threshold metric={metric_name}")
            continue
        if value is None:
            failures.append(f"{prefix} missing_metric metric={metric_name}")
            continue
        if value < threshold:
            failures.append(
                f"{prefix} metric_gate_failed {metric_name}={value:.6f} < min_required={threshold:.6f}"
            )

    return len(failures) == 0, failures, results


def _evaluate_metric_gates(metrics: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(policy.get("enabled", True))
    minimums = dict(policy.get("minimums") or {})

    if not enabled:
        return {
            "enabled": False,
            "passed": True,
            "skipped": True,
            "thresholds": minimums,
            "results": {},
            "failures": [],
        }

    passed, failures, results = _evaluate_minimums(metrics, minimums, prefix="metrics")
    return {
        "enabled": True,
        "passed": passed,
        "skipped": False,
        "thresholds": minimums,
        "results": results,
        "failures": failures,
    }


def _evaluate_drift_gates(policy: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(policy.get("enabled", True))
    report_path = Path(str(policy.get("report_path") or "")).expanduser()
    max_abs_delta = dict(policy.get("max_abs_delta") or {})

    detail = {
        "enabled": enabled,
        "report_path": str(report_path),
        "thresholds": max_abs_delta,
        "results": {},
        "passed": False,
        "skipped": False,
        "failures": [],
    }

    if not enabled:
        detail["passed"] = True
        detail["skipped"] = True
        return detail

    if not report_path.exists():
        detail["failures"].append(f"drift report_not_found path={report_path}")
        return detail

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        detail["failures"].append(f"drift report_unreadable path={report_path}")
        return detail

    if not isinstance(payload, dict):
        detail["failures"].append(f"drift report_invalid_json path={report_path}")
        return detail

    for field, threshold_raw in max_abs_delta.items():
        threshold = _safe_float(threshold_raw)
        delta = _safe_float(payload.get(field))
        abs_delta = abs(delta) if delta is not None else None
        passed = threshold is not None and abs_delta is not None and abs_delta <= threshold

        detail["results"][field] = {
            "delta": delta,
            "abs_delta": abs_delta,
            "max_allowed_abs": threshold,
            "passed": passed,
        }

        if threshold is None:
            detail["failures"].append(f"drift invalid_threshold field={field}")
            continue
        if delta is None:
            detail["failures"].append(f"drift missing_field field={field}")
            continue
        if abs_delta > threshold:
            detail["failures"].append(
                f"drift gate_failed {field}={delta:.6f} (abs max={threshold:.6f})"
            )

    detail["passed"] = len(detail["failures"]) == 0
    return detail


def _evaluate_generalization_gates(policy: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(policy.get("enabled", False))
    report_path = Path(str(policy.get("report_path") or "")).expanduser()
    required_scenarios = dict(policy.get("required_scenarios") or {})

    detail = {
        "enabled": enabled,
        "report_path": str(report_path),
        "required_scenarios": required_scenarios,
        "results": {},
        "passed": False,
        "skipped": False,
        "failures": [],
    }

    if not enabled:
        detail["passed"] = True
        detail["skipped"] = True
        return detail

    if not required_scenarios:
        detail["passed"] = True
        detail["skipped"] = True
        return detail

    if not report_path.exists():
        detail["failures"].append(f"generalization report_not_found path={report_path}")
        return detail

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        detail["failures"].append(f"generalization report_unreadable path={report_path}")
        return detail

    if not isinstance(payload, dict):
        detail["failures"].append(f"generalization report_invalid_json path={report_path}")
        return detail

    scenarios = payload.get("scenarios") or {}
    if not isinstance(scenarios, dict):
        detail["failures"].append("generalization report_missing_scenarios_object")
        return detail

    for scenario_name, minimums in required_scenarios.items():
        scenario_metrics = scenarios.get(scenario_name)
        if not isinstance(scenario_metrics, dict):
            detail["results"][scenario_name] = {
                "present": False,
                "passed": False,
                "results": {},
            }
            detail["failures"].append(f"generalization missing_scenario scenario={scenario_name}")
            continue

        passed, failures, results = _evaluate_minimums(
            scenario_metrics,
            dict(minimums or {}),
            prefix=f"generalization[{scenario_name}]",
        )
        detail["results"][scenario_name] = {
            "present": True,
            "passed": passed,
            "results": results,
        }
        detail["failures"].extend(failures)

    detail["passed"] = len(detail["failures"]) == 0
    return detail


def build_promotion_gate_report(*, metrics: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    metric_gates = _evaluate_metric_gates(metrics, dict(policy.get("metrics") or {}))
    drift_gates = _evaluate_drift_gates(dict(policy.get("drift") or {}))
    generalization_gates = _evaluate_generalization_gates(dict(policy.get("generalization") or {}))

    failures = []
    failures.extend(metric_gates.get("failures", []))
    failures.extend(drift_gates.get("failures", []))
    failures.extend(generalization_gates.get("failures", []))

    passed = (
        bool(metric_gates.get("passed", False))
        and bool(drift_gates.get("passed", False))
        and bool(generalization_gates.get("passed", False))
    )

    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "failures": failures,
        "metric_gates": metric_gates,
        "drift_gates": drift_gates,
        "generalization_gates": generalization_gates,
    }
