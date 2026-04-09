"""Optional Platt-style calibration for CV skill confidence.

After labeling around 20-30 CVs, fit `a` and `b` so
`sigmoid(a * raw + b)` better matches empirical precision.

Config file (either path):
- Environment: `CONFIDENCE_CALIBRATION_PATH` -> full path to a JSON file
- Default: `skillmatch-pro-back/artifacts/confidence_calibration.json`

Schema:
  {"a": 4.0, "b": -2.0}
  {"disabled": true}
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

_DEFAULT_ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "confidence_calibration.json"


def _clip_unit(raw: float) -> float:
    return max(0.001, min(0.999, float(raw)))


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _binary_logloss_from_probs(probs: list[float], labels: list[int]) -> float:
    if not probs:
        return 0.0
    eps = 1e-12
    total = 0.0
    for p, y in zip(probs, labels):
        q = max(eps, min(1.0 - eps, float(p)))
        total += -(y * math.log(q) + (1 - y) * math.log(1.0 - q))
    return total / len(probs)


def _brier_from_probs(probs: list[float], labels: list[int]) -> float:
    if not probs:
        return 0.0
    total = 0.0
    for p, y in zip(probs, labels):
        d = float(p) - float(y)
        total += d * d
    return total / len(probs)


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
    x = _clip_unit(raw)
    z = float(a) * x + float(b)
    out = _sigmoid(z)
    return max(0.01, min(0.99, float(out)))


def calibrate_confidence_if_configured(raw: float) -> float:
    a, b, on = load_platt_params()
    if not on:
        return float(raw)
    return apply_platt_on_unit_interval(raw, a, b)


def fit_platt_params(
    raw_scores: list[float],
    labels: list[int],
    *,
    l2: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> tuple[float, float, dict[str, float]]:
    """Fit Platt scaling parameters with Newton updates on one feature.

    Returns `(a, b, metrics)` where metrics include train log-loss/Brier
    before and after fitting.
    """
    if len(raw_scores) != len(labels):
        raise ValueError("raw_scores and labels must have same length")
    if not raw_scores:
        raise ValueError("empty calibration dataset")

    xs = [_clip_unit(x) for x in raw_scores]
    ys: list[int] = []
    for y in labels:
        yi = int(y)
        if yi not in (0, 1):
            raise ValueError("labels must be binary (0/1)")
        ys.append(yi)

    positives = sum(ys)
    negatives = len(ys) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("calibration needs both positive and negative samples")

    a = 1.0
    b = 0.0
    baseline_probs = [x for x in xs]
    before_logloss = _binary_logloss_from_probs(baseline_probs, ys)
    before_brier = _brier_from_probs(baseline_probs, ys)

    def _loss(pa: float, pb: float) -> float:
        probs = [_sigmoid(pa * x + pb) for x in xs]
        return _binary_logloss_from_probs(probs, ys) + 0.5 * float(l2) * (pa * pa)

    prev = _loss(a, b)
    iters = 0

    for it in range(int(max_iter)):
        iters = it + 1
        probs = [_sigmoid(a * x + b) for x in xs]
        w = [max(1e-9, p * (1.0 - p)) for p in probs]

        g_a = sum((p - y) * x for p, y, x in zip(probs, ys, xs)) + float(l2) * a
        g_b = sum((p - y) for p, y in zip(probs, ys))

        h_aa = sum(wi * x * x for wi, x in zip(w, xs)) + float(l2)
        h_ab = sum(wi * x for wi, x in zip(w, xs))
        h_bb = sum(w)

        det = (h_aa * h_bb) - (h_ab * h_ab)
        if abs(det) < 1e-12:
            break

        step_a = (h_bb * g_a - h_ab * g_b) / det
        step_b = (-h_ab * g_a + h_aa * g_b) / det

        step = 1.0
        improved = False
        while step >= 1e-4:
            cand_a = a - step * step_a
            cand_b = b - step * step_b
            cand_loss = _loss(cand_a, cand_b)
            if cand_loss <= prev:
                a, b = cand_a, cand_b
                prev = cand_loss
                improved = True
                break
            step *= 0.5

        if not improved:
            break
        if (abs(step * step_a) + abs(step * step_b)) < float(tol):
            break

    after_probs = [apply_platt_on_unit_interval(x, a, b) for x in xs]
    after_logloss = _binary_logloss_from_probs(after_probs, ys)
    after_brier = _brier_from_probs(after_probs, ys)

    metrics = {
        "iterations": float(iters),
        "before_logloss": float(before_logloss),
        "after_logloss": float(after_logloss),
        "before_brier": float(before_brier),
        "after_brier": float(after_brier),
    }
    return float(a), float(b), metrics
