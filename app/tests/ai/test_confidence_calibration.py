from pathlib import Path

from app.ai.confidence_calibration import apply_platt_on_unit_interval, load_platt_params


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
