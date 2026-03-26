from app.scripts.summarize_whylogs_profiles import _daily_aggregate, _daily_drift_aggregate


def test_daily_summary_flags_calibration_warning():
    rows = [
        {
            "file_name": "profile_a.bin",
            "job_title": "data analyst",
            "timestamp_utc": "2026-03-25T10:00:00Z",
            "day_utc": "2026-03-25",
            "n": 100,
            "mean": 0.97,
            "stddev": 0.02,
            "min": 0.6,
            "max": 1.0,
            "q10": 0.9,
            "median": 0.98,
            "q90": 1.0,
        }
    ]

    out = _daily_aggregate(
        rows,
        drift_by_day={},
        mean_warning_threshold=0.95,
        std_warning_threshold=0.03,
        retrain_psi_threshold=0.2,
        retrain_calibration_threshold=0.05,
    )

    assert len(out) == 1
    assert out[0]["calibration_warning"] is True
    assert out[0]["retrain_trigger"] is False


def test_daily_summary_sets_retrain_trigger_from_drift():
    rows = [
        {
            "file_name": "profile_b.bin",
            "job_title": "ml engineer",
            "timestamp_utc": "2026-03-26T10:00:00Z",
            "day_utc": "2026-03-26",
            "n": 200,
            "mean": 0.7,
            "stddev": 0.15,
            "min": 0.1,
            "max": 0.95,
            "q10": 0.3,
            "median": 0.72,
            "q90": 0.9,
        }
    ]
    drift_by_day = _daily_drift_aggregate(
        [
            {"day_utc": "2026-03-26", "score_psi": 0.25, "calibration_drift": 0.01},
            {"day_utc": "2026-03-26", "score_psi": 0.11, "calibration_drift": 0.07},
        ]
    )

    out = _daily_aggregate(
        rows,
        drift_by_day=drift_by_day,
        mean_warning_threshold=0.95,
        std_warning_threshold=0.03,
        retrain_psi_threshold=0.2,
        retrain_calibration_threshold=0.05,
    )

    assert len(out) == 1
    assert out[0]["score_psi_max"] == 0.25
    assert out[0]["calibration_drift_abs_max"] == 0.07
    assert out[0]["retrain_trigger"] is True
    assert "score_psi" in out[0]["retrain_trigger_reasons"]
    assert "calibration_drift" in out[0]["retrain_trigger_reasons"]
