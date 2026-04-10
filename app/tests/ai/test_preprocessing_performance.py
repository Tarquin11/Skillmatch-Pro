from app.ai.preprocessing import normalize_performance


def test_normalize_performance_supports_common_hr_labels():
    assert normalize_performance("Fully Meets") > 0.0
    assert normalize_performance("Good") > 0.0
    assert normalize_performance("Needs Improvement") > 0.0
    assert normalize_performance("Good") > normalize_performance("Needs Improvement")
