import json
import sys
from pathlib import Path
import uuid

from app.scripts import evaluate_cv_fairness


def _valid_payload(*, degraded: bool = False, empty_text: bool = False) -> dict:
    warnings = ["empty_text"] if empty_text else []
    return {
        "filename": "x.pdf",
        "ok": True,
        "degraded": degraded,
        "errors": [],
        "warnings": warnings,
        "text_length": 0 if empty_text else 42,
        "skills": ["python"],
        "skills_grouped": {"technical": ["python"], "management": [], "business": [], "soft-skills": [], "other": []},
        "skill_hierarchy": [],
        "skill_graph": {},
        "extracted_languages": [],
        "language_details": [],
        "extraction_channels": {
            "catalog_match": ["python"],
            "open_vocab": [],
            "soft_skill": [],
            "sentence": [],
            "semantic_augment": [],
            "language": [],
            "project_text": [],
        },
        "preview": "test",
        "extracted_skills": [
            {"skill": "python", "confidence": 0.9, "confidence_normalized": 1.0, "source": "exact", "evidence": []}
        ],
        "predicted_title": "Developer",
        "predicted_experience_years": 2.0,
    }


def _local_tmp_dir() -> Path:
    root = Path("artifacts") / "pytest_local" / f"fairness_eval_{uuid.uuid4().hex[:10]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_evaluate_cv_fairness_generates_group_disparities(monkeypatch):
    root = _local_tmp_dir()
    cv_a1 = root / "a1.pdf"
    cv_a2 = root / "a2.pdf"
    cv_b1 = root / "b1.pdf"
    cv_b2 = root / "b2.pdf"
    for path in (cv_a1, cv_a2, cv_b1, cv_b2):
        path.write_bytes(b"fake")

    labels = root / "labels.jsonl"
    labels.write_text(
        "\n".join(
            [
                json.dumps({"path": str(cv_a1), "identity": {"industry": "finance"}}),
                json.dumps({"path": str(cv_a2), "identity": {"industry": "finance"}}),
                json.dumps({"path": str(cv_b1), "identity": {"industry": "health"}}),
                json.dumps({"path": str(cv_b2), "identity": {"industry": "health"}}),
            ]
        ),
        encoding="utf-8",
    )

    def _fake_parse(**kwargs):
        filename = str(kwargs.get("filename", ""))
        if filename == "a2.pdf":
            raise RuntimeError("boom")
        return _valid_payload(degraded=(filename == "b2.pdf"))

    monkeypatch.setattr(evaluate_cv_fairness, "parse_cv_safe", _fake_parse)

    out = root / "cv_fairness_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_cv_fairness",
            "--labels-jsonl",
            str(labels),
            "--group-field",
            "identity.industry",
            "--min-group-size",
            "1",
            "--out",
            str(out),
        ],
    )
    evaluate_cv_fairness.main()

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["totals"]["records"] == 4
    assert report["totals"]["eligible_group_count"] == 2
    assert report["groups"]["finance"]["totals"]["crashes"] == 1
    assert report["groups"]["health"]["totals"]["crashes"] == 0
    assert report["disparities"]["crash_rate_gap"] == 0.5
    assert report["disparities"]["schema_valid_rate_gap"] == 0.5
