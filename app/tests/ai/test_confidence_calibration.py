from pathlib import Path
import math
import random

import pytest

from app.ai.confidence_calibration import apply_platt_on_unit_interval, fit_platt_params, load_platt_params


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_load_platt_params_disabled_without_file(monkeypatch):
    monkeypatch.delenv("CONFIDENCE_CALIBRATION_PATH", raising=False)
    a, b, on = load_platt_params()
    assert on is False
    assert a == 1.0 and b == 0.0


def test_load_platt_params_from_env_path(monkeypatch):
    p = _repo_root() / ".pytest_confidence_calib_tmp.json"
    try:
        p.write_text('{"a": 6.0, "b": -3.0}', encoding="utf-8")
        monkeypatch.setenv("CONFIDENCE_CALIBRATION_PATH", str(p))
        a, b, on = load_platt_params()
        assert on is True
        assert a == 6.0 and b == -3.0
        y = apply_platt_on_unit_interval(0.72, a, b)
        assert 0.01 < y < 0.99
    finally:
        p.unlink(missing_ok=True)


def test_load_platt_respects_disabled_flag(monkeypatch):
    p = _repo_root() / ".pytest_confidence_calib_tmp.json"
    try:
        p.write_text('{"disabled": true, "a": 9.0, "b": 0.0}', encoding="utf-8")
        monkeypatch.setenv("CONFIDENCE_CALIBRATION_PATH", str(p))
        _a, _b, on = load_platt_params()
        assert on is False
    finally:
        p.unlink(missing_ok=True)


def test_fit_platt_params_requires_both_classes():
    with pytest.raises(ValueError):
        fit_platt_params([0.2, 0.4, 0.9], [1, 1, 1])


def test_fit_platt_params_improves_logloss_on_shifted_data():
    rng = random.Random(42)
    raws: list[float] = []
    ys: list[int] = []

    for _ in range(120):
        x = rng.uniform(0.35, 0.99)
        raws.append(x)
        # Strongly over-confident synthetic raw scores: true positive rate is lower.
        p_true = 1.0 / (1.0 + math.exp(-(6.0 * (x - 0.78))))
        ys.append(1 if rng.random() < p_true else 0)

    baseline = []
    for x in raws:
        q = max(1e-9, min(1.0 - 1e-9, x))
        baseline.append(q)
    before = -sum(y * math.log(p) + (1 - y) * math.log(1 - p) for p, y in zip(baseline, ys)) / len(ys)

    a, b, metrics = fit_platt_params(raws, ys)
    calibrated = [apply_platt_on_unit_interval(x, a, b) for x in raws]
    after = -sum(y * math.log(p) + (1 - y) * math.log(1 - p) for p, y in zip(calibrated, ys)) / len(ys)

    assert after <= before
    assert metrics["after_logloss"] <= metrics["before_logloss"]
