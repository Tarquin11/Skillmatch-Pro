from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from app.services.promotion_gates import build_promotion_gate_report, load_gate_policy


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _latest_file(glob_pattern: str) -> Path | None:
    files = list(Path(".").glob(glob_pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _extract_model_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "roc_auc",
        "f1",
        "map_at_k",
        "average_precision",
        "precision",
        "recall",
        "precision_at_k",
        "recall_at_k",
        "ndcg_at_k",
    ]
    return {k: payload.get(k) for k in keys if k in payload}


def _evaluate_model_gate(
    metrics_path: Path,
    policy_path: Path,
) -> dict[str, Any]:
    payload = _safe_load_json(metrics_path)

    embedded_gate = payload.get("promotion_gate")
    if isinstance(embedded_gate, dict) and "passed" in embedded_gate:
        return {
            "source": "embedded_promotion_gate",
            "metrics_path": str(metrics_path),
            "passed": bool(embedded_gate.get("passed", False)),
            "failures": list(embedded_gate.get("failures") or []),
            "detail": embedded_gate,
        }

    policy = load_gate_policy(str(policy_path))
    computed_gate = build_promotion_gate_report(
        metrics=_extract_model_metrics(payload),
        policy=policy,
    )
    return {
        "source": "computed_from_policy",
        "metrics_path": str(metrics_path),
        "passed": bool(computed_gate.get("passed", False)),
        "failures": list(computed_gate.get("failures") or []),
        "detail": computed_gate,
    }


def _evaluate_scheduled_gate(
    gate_path: Path,
    max_age_hours: float | None,
) -> dict[str, Any]:
    payload = _safe_load_json(gate_path)
    passed = bool(payload.get("passed", False))
    failures = list(payload.get("failures") or [])

    mtime = datetime.fromtimestamp(gate_path.stat().st_mtime, tz=timezone.utc)
    age_hours = (_utc_now() - mtime).total_seconds() / 3600.0

    stale = False
    if max_age_hours is not None and age_hours > max_age_hours:
        stale = True
        passed = False
        failures.append(
            f"scheduled_gate_stale age_hours={age_hours:.2f} > max_age_hours={max_age_hours:.2f}"
        )

    return {
        "gate_path": str(gate_path),
        "passed": passed,
        "failures": failures,
        "mtime_utc": mtime.isoformat(),
        "age_hours": round(age_hours, 4),
        "stale": stale,
        "detail": payload,
    }


def _run_tests(
    pytest_targets: list[str],
    log_path: Path,
    pytest_basetemp_root: str | None = None,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    base_root = Path(pytest_basetemp_root).expanduser() if pytest_basetemp_root else Path(tempfile.gettempdir())
    base_temp = base_root / f"skillmatch_pytest_release_{uuid.uuid4().hex[:8]}"
    base_temp.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *pytest_targets,
        "-q",
        "--basetemp",
        str(base_temp),
    ]

    started = perf_counter()
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    elapsed = perf_counter() - started

    return {
        "command": " ".join(cmd),
        "passed": proc.returncode == 0,
        "return_code": proc.returncode,
        "duration_seconds": round(elapsed, 3),
        "log_path": str(log_path),
        "base_temp": str(base_temp),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Release readiness gate: tests must pass and latest gate reports must be green."
    )
    parser.add_argument(
        "--pytest-target",
        action="append",
        dest="pytest_targets",
        help="Pytest target path/pattern (repeatable). Default: app/tests",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest execution (not recommended for final release decision).",
    )
    parser.add_argument(
        "--policy",
        default="app/config/promotion_policy.json",
        help="Promotion gate policy path.",
    )
    parser.add_argument(
        "--metrics-glob",
        default="artifacts/matcher_metrics_*.json",
        help="Glob for selecting latest model metrics file.",
    )
    parser.add_argument(
        "--scheduled-gate-glob",
        default="artifacts/evaluations/generalization_gate_*.json",
        help="Glob for selecting latest scheduled generalization gate file.",
    )
    parser.add_argument(
        "--max-scheduled-gate-age-hours",
        type=float,
        default=36.0,
        help="Fail if scheduled gate report is older than this age in hours.",
    )
    parser.add_argument(
        "--out",
        default="artifacts/release_readiness.json",
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--pytest-log",
        default="artifacts/release_pytest.log",
        help="Path where pytest stdout/stderr is written.",
    )
    parser.add_argument(
        "--pytest-basetemp",
        default=None,
        help="Optional root directory for pytest --basetemp (useful on Windows when default temp is locked).",
    )
    args = parser.parse_args()

    pytest_targets = args.pytest_targets or ["app/tests"]
    policy_path = Path(args.policy).expanduser()
    metrics_path = _latest_file(args.metrics_glob)
    scheduled_gate_path = _latest_file(args.scheduled_gate_glob)

    summary: dict[str, Any] = {
        "generated_at_utc": _utc_now().isoformat(),
        "ready": False,
        "tests": {},
        "gates": {
            "model": {},
            "scheduled_generalization": {},
        },
        "failures": [],
    }

    if args.skip_tests:
        summary["tests"] = {
            "skipped": True,
            "passed": False,
            "reason": "skip_tests_enabled",
        }
        summary["failures"].append("tests skipped")
    else:
        tests = _run_tests(
            pytest_targets,
            Path(args.pytest_log).expanduser(),
            pytest_basetemp_root=args.pytest_basetemp,
        )
        summary["tests"] = tests
        if not tests["passed"]:
            summary["failures"].append("tests failed")

    if not policy_path.exists():
        summary["gates"]["model"] = {"passed": False, "error": f"policy_not_found path={policy_path}"}
        summary["failures"].append(f"policy_not_found path={policy_path}")
    elif metrics_path is None:
        summary["gates"]["model"] = {"passed": False, "error": "metrics_report_not_found"}
        summary["failures"].append("metrics_report_not_found")
    else:
        model_gate = _evaluate_model_gate(metrics_path, policy_path)
        summary["gates"]["model"] = model_gate
        if not model_gate["passed"]:
            summary["failures"].append("model_gate_failed")

    if scheduled_gate_path is None:
        summary["gates"]["scheduled_generalization"] = {
            "passed": False,
            "error": "scheduled_generalization_gate_not_found",
        }
        summary["failures"].append("scheduled_generalization_gate_not_found")
    else:
        scheduled_gate = _evaluate_scheduled_gate(
            scheduled_gate_path,
            max_age_hours=float(args.max_scheduled_gate_age_hours)
            if args.max_scheduled_gate_age_hours is not None
            else None,
        )
        summary["gates"]["scheduled_generalization"] = scheduled_gate
        if not scheduled_gate["passed"]:
            summary["failures"].append("scheduled_generalization_gate_failed")

    tests_passed = bool(summary["tests"].get("passed", False))
    model_gate_passed = bool(summary["gates"]["model"].get("passed", False))
    scheduled_gate_passed = bool(summary["gates"]["scheduled_generalization"].get("passed", False))
    summary["ready"] = tests_passed and model_gate_passed and scheduled_gate_passed

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["ready"] else 2)


if __name__ == "__main__":
    main()
