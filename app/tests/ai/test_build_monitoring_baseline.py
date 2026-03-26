import pytest

from app.scripts.build_monitoring_baseline import _quality_stats, _validate_quality


def test_quality_gate_rejects_saturated_distribution():
    stats = _quality_stats([1.0] * 1000)
    with pytest.raises(ValueError):
        _validate_quality(
            stats,
            min_stddev=0.01,
            min_unique_scores=50,
            max_dominant_ratio=0.995,
        )


def test_quality_gate_accepts_diverse_distribution():
    stats = _quality_stats([i / 1000 for i in range(1000)])
    _validate_quality(
        stats,
        min_stddev=0.01,
        min_unique_scores=50,
        max_dominant_ratio=0.995,
    )
