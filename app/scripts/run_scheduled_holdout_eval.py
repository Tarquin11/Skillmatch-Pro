from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from app.services.promotion_gates import build_promotion_gate_report, load_gate_policy

def _parse_scenarios(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid --scenario format: {raw} (expected name=path)")
        name, path = raw.split("=", 1)
        name = name.strip()
        p = Path(path.strip())
        if not name:
            raise ValueError(f"Invalid scenario name in: {raw}")
        out[name] = p
    return out

def _assert_fresh(path: Path, max_age_hours: float) -> dict[str, float | str]:
    if not path.exists():
        raise FileNotFoundError(f"Holdout scenario file not found: {path}")

    now = datetime.now(timezone.utc)
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age_hours = (now - mtime).total_seconds() / 3600.0

    if age_hours > max_age_hours:
        raise RuntimeError(
            f"Holdout file is stale: {path} age_hours={age_hours:.2f} > max_age_hours={max_age_hours:.2f}"
        )

    return {
        "path": str(path),
        "mtime_utc": mtime.isoformat(),
        "age_hours": round(age_hours, 4),
    }
def main() -> None:
    parser = argparse.ArgumentParser(description="Scheduled evaluation on fresh holdout scenarios.")
    parser.add_argument("--model", default="artifacts/matcher.joblib")
    parser.add_argument("--scenario", action="append", required=True, help="name=path_to_holdout")
    parser.add_argument("--out-dir", default="artifacts/evaluations")
    parser.add_argument("--latest-out", default="artifacts/generalization_report.json")
    parser.add_argument("--policy", default="app/config/promotion_policy.json")
    parser.add_argument("--max-holdout-age-hours", type=float, default=36.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-semantic", action="store_true", help="Disable semantic embeddings for holdout evaluation.")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    scenarios = _parse_scenarios(args.scenario)
    freshness: dict[str, dict[str, float | str]] = {}
    for name, path in scenarios.items():
        freshness[name] = _assert_fresh(path, max_age_hours=float(args.max_holdout_age_hours))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    versioned_report = out_dir / f"generalization_report_{stamp}.json"
    latest_report = Path(args.latest_out)

    cmd = [
        sys.executable,
        "-m",
        "app.scripts.evaluate_generalization",
        "--model",
        str(model_path),
        "--k",
        str(int(args.k)),
        "--threshold",
        str(float(args.threshold)),
        "--out",
        str(versioned_report),
    ]
    if args.no_semantic:
        cmd.append("--no-semantic")
    for name, path in scenarios.items():
        cmd.extend(["--scenario", f"{name}={path}"])

    subprocess.run(cmd, check=True)

    payload = json.loads(versioned_report.read_text(encoding="utf-8"))
    payload["scheduled_eval"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "policy_path": str(Path(args.policy)),
        "max_holdout_age_hours": float(args.max_holdout_age_hours),
        "freshness": freshness,
    }
    versioned_report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    latest_report.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(versioned_report, latest_report)

    # Evaluate only generalization gates during scheduled eval.
    policy = load_gate_policy(
        args.policy,
        overrides={
            "metrics": {"enabled": False},
            "drift": {"enabled": False},
            "generalization": {"report_path": str(latest_report)},
            "robustness": {"enabled": False},
        },
    )
    gate = build_promotion_gate_report(metrics={}, policy=policy)

    gate_out = out_dir / f"generalization_gate_{stamp}.json"
    gate_out.write_text(json.dumps(gate, indent=2), encoding="utf-8")

    summary = {
        "generalization_report": str(versioned_report),
        "latest_report": str(latest_report),
        "gate_report": str(gate_out),
        "gate_passed": bool(gate.get("passed", False)),
        "gate_failures": list(gate.get("failures", [])),
    }
    print(json.dumps(summary, indent=2))
    if not gate.get("passed", False):
        raise SystemExit(2)
    
if __name__ == "__main__":
    main()
