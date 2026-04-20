import json
from pathlib import Path
import uuid

from app.services.promotion_gates import build_promotion_gate_report


def _base_policy(report_path: Path):
    return {
        "metrics": {"enabled": False},
        "drift": {"enabled": False},
        "generalization": {"enabled": False},
        "robustness": {"enabled": False},
        "fairness": {
            "enabled": True,
            "report_path": str(report_path),
            "minimum_group_count": 2,
            "max_abs_gaps": {
                "schema_valid_rate_gap": 0.10,
                "crash_rate_gap": 0.05,
                "degraded_rate_gap": 0.20,
                "empty_text_rate_gap": 0.10,
                "ok_rate_gap": 0.10,
            },
            "tracked": [
                "schema_valid_rate_gap",
                "crash_rate_gap",
            ],
        },
    }


def _local_tmp_dir() -> Path:
    root = Path("artifacts") / "pytest_local" / f"fairness_gate_{uuid.uuid4().hex[:10]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_fairness_gates_pass_when_gaps_within_thresholds():
    root = _local_tmp_dir()
    report_path = root / "cv_fairness_report.json"
    report_path.write_text(
        json.dumps(
            {
                "totals": {"eligible_group_count": 2},
                "disparities": {
                    "schema_valid_rate_gap": 0.03,
                    "crash_rate_gap": 0.01,
                    "degraded_rate_gap": 0.08,
                    "empty_text_rate_gap": 0.02,
                    "ok_rate_gap": 0.04,
                },
            }
        ),
        encoding="utf-8",
    )
    gate = build_promotion_gate_report(metrics={}, policy=_base_policy(report_path))
    assert gate["passed"] is True
    assert gate["fairness_gates"]["passed"] is True
    assert gate["fairness_gates"]["tracked"]["schema_valid_rate_gap"] == 0.03
    assert gate["fairness_gates"]["tracked"]["crash_rate_gap"] == 0.01


def test_fairness_gates_fail_when_gap_exceeds_threshold():
    root = _local_tmp_dir()
    report_path = root / "cv_fairness_report.json"
    report_path.write_text(
        json.dumps(
            {
                "totals": {"eligible_group_count": 2},
                "disparities": {
                    "schema_valid_rate_gap": 0.12,
                    "crash_rate_gap": 0.07,
                    "degraded_rate_gap": 0.08,
                    "empty_text_rate_gap": 0.02,
                    "ok_rate_gap": 0.04,
                },
            }
        ),
        encoding="utf-8",
    )
    gate = build_promotion_gate_report(metrics={}, policy=_base_policy(report_path))
    assert gate["passed"] is False
    assert gate["fairness_gates"]["passed"] is False
    failures = gate["fairness_gates"]["failures"]
    assert any("schema_valid_rate_gap" in msg for msg in failures)
    assert any("crash_rate_gap" in msg for msg in failures)


def test_fairness_gates_fail_on_insufficient_groups():
    root = _local_tmp_dir()
    report_path = root / "cv_fairness_report.json"
    report_path.write_text(
        json.dumps(
            {
                "totals": {"eligible_group_count": 1},
                "disparities": {
                    "schema_valid_rate_gap": 0.0,
                    "crash_rate_gap": 0.0,
                    "degraded_rate_gap": 0.0,
                    "empty_text_rate_gap": 0.0,
                    "ok_rate_gap": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    gate = build_promotion_gate_report(metrics={}, policy=_base_policy(report_path))
    assert gate["passed"] is False
    assert gate["fairness_gates"]["passed"] is False
    failures = gate["fairness_gates"]["failures"]
    assert any("insufficient_groups" in msg for msg in failures)
