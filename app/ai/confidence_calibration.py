"""Optional Platt-style calibration: map heuristic confidence to a probability-like score.

After labeling ~20–30 CVs (skill present / absent vs your raw score), fit
`a` and `b` so that `sigmoid(a * raw + b)` matches empirical precision, then
write JSON (see below). If no file is present, scores are unchanged.

Config file (either path):
- Environment: `CONFIDENCE_CALIBRATION_PATH` → full path to a JSON file
- Default: `skillmatch-pro-back/artifacts/confidence_calibration.json`

Schema:
  {"a": 4.0, "b": -2.0}   # calibrated = sigmoid(a * raw + b), raw in ~[0,1]
  {"disabled": true}      # turn off without deleting the file
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

_DEFAULT_ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "confidence_calibration.json"


def _artifact_path() -> Path | None:
    env = (os.environ.get("CONFIDENCE_CALIBRATION_PATH") or "").strip()
    if env:
        return Path(env)
    if _DEFAULT_ARTIFACT.is_file():
        return _DEFAULT_ARTIFACT
    return None


def load_platt_params() -> tuple[float, float, bool]:
    """Return (a, b, enabled). When enabled is False, skip calibration."""
    path = _artifact_path()
    if path is None or not path.is_file():
        return 1.0, 0.0, False
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if data.get("disabled") is True:
            return 1.0, 0.0, False
        a = float(data.get("a", 1.0))
        b = float(data.get("b", 0.0))
        return a, b, True
    except Exception:
        return 1.0, 0.0, False


def apply_platt_on_unit_interval(raw: float, a: float, b: float) -> float:
    """calibrated = sigmoid(a * raw + b), with raw on ~[0,1] from heuristics."""
    x = max(0.001, min(0.999, float(raw)))
    z = float(a) * x + float(b)
    out = 1.0 / (1.0 + math.exp(-z))
    return max(0.01, min(0.99, float(out)))


def calibrate_confidence_if_configured(raw: float) -> float:
    a, b, on = load_platt_params()
    if not on:
        return float(raw)
    return apply_platt_on_unit_interval(raw, a, b)
