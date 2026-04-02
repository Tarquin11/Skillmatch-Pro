import json
from pathlib import Path
import uuid

from app.services.promotion_gates import build_promotion_gate_report


def _base_policy(report_path: Path):
    return {
        "metrics": {"enabled": False},
        "drift": {"enabled": False},
        "generalization": {"enabled": False},
        "robustness": {
            "enabled": True,
            "report_path": str(report_path),
            "minimums": {"schema_valid_rate": 1.0},
            "maximums": {"crash_rate": 0.0},
            "advisory_maximums": {"degraded_rate": 0.3},
            "tracked": ["degraded_rate", "empty_text_rate"],
        },
    }


def _local_tmp_dir() -> Path:
    root = Path("artifacts") / "pytest_local" / f"robust_gate_{uuid.uuid4().hex[:10]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_robustness_gates_pass_when_targets_met():
    root = _local_tmp_dir()
    report_path = root / "cv_robustness_report.json"
    report_path.write_text(
        json.dumps(
            {
                "kpis": {
                    "crash_rate": 0.0,
                    "schema_valid_rate": 1.0,
                    "degraded_rate": 0.25,
                    "empty_text_rate": 0.02,
                }
            }
        ),
        encoding="utf-8",
    )
    gate = build_promotion_gate_report(metrics={}, policy=_base_policy(report_path))
    assert gate["passed"] is True
    assert gate["robustness_gates"]["passed"] is True
    assert gate["robustness_gates"]["tracked"]["degraded_rate"] == 0.25
    assert gate["robustness_gates"]["tracked"]["empty_text_rate"] == 0.02


def test_robustness_gates_fail_on_crash_or_schema():
    root = _local_tmp_dir()
    report_path = root / "cv_robustness_report.json"
    report_path.write_text(
        json.dumps(
            {
                "kpis": {
                    "crash_rate": 0.01,
                    "schema_valid_rate": 0.95,
                    "degraded_rate": 0.40,
                    "empty_text_rate": 0.10,
                }
            }
        ),
        encoding="utf-8",
    )
    gate = build_promotion_gate_report(metrics={}, policy=_base_policy(report_path))
    assert gate["passed"] is False
    assert gate["robustness_gates"]["passed"] is False
    failures = gate["robustness_gates"]["failures"]
    assert any("crash_rate" in msg for msg in failures)
    assert any("schema_valid_rate" in msg for msg in failures)


def test_robustness_advisory_threshold_does_not_fail_gate():
    root = _local_tmp_dir()
    report_path = root / "cv_robustness_report.json"
    report_path.write_text(
        json.dumps(
            {
                "kpis": {
                    "crash_rate": 0.0,
                    "schema_valid_rate": 1.0,
                    "degraded_rate": 0.91,
                    "empty_text_rate": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )
    gate = build_promotion_gate_report(metrics={}, policy=_base_policy(report_path))
    assert gate["passed"] is True
    assert gate["robustness_gates"]["passed"] is True
    advisories = gate["robustness_gates"]["advisories"]
    assert any("degraded_rate" in msg for msg in advisories)
