from types import SimpleNamespace

from app.ai.preprocessing import preprocess_employee


def test_preprocess_employee_uses_neutral_defaults_for_missing_hr_metrics():
    employee = SimpleNamespace(
        id=1,
        first_name="Jane",
        last_name="Doe",
        full_name="Jane Doe",
        department="IT",
        position="web developer",
        hire_date=None,
        termination_date=None,
        employment_status="",
        skills=[],
        performance_score=None,
        engagement_survey=None,
        emp_satisfaction=None,
    )

    row = preprocess_employee(employee)
    assert row["performance_score"] == 0.5
    assert row["engagement_score"] == 0.5
    assert row["satisfaction_score"] == 0.5
