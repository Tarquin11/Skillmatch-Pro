import json
import sys
from pathlib import Path
import uuid

from app.scripts import evaluate_cv_robustness


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
    root = Path("artifacts") / "pytest_local" / f"robust_eval_{uuid.uuid4().hex[:10]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_evaluate_cv_robustness_generates_expected_kpis(monkeypatch):
    root = _local_tmp_dir()
    cv_a = root / "a.pdf"
    cv_b = root / "b.pdf"
    cv_a.write_bytes(b"fake")
    cv_b.write_bytes(b"fake")

    labels = root / "labels.jsonl"
    labels.write_text(
        "\n".join(
            [
                json.dumps({"path": str(cv_a), "expected_skills": ["python"]}),
                json.dumps({"path": str(cv_b), "expected_skills": ["sql"]}),
            ]
        ),
        encoding="utf-8",
    )

    calls = {"n": 0}

    def _fake_parse(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _valid_payload(degraded=False, empty_text=False)
        raise RuntimeError("boom")

    monkeypatch.setattr(evaluate_cv_robustness, "parse_cv_safe", _fake_parse)

    out = root / "cv_robustness_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_cv_robustness",
            "--labels-jsonl",
            str(labels),
            "--out",
            str(out),
        ],
    )
    evaluate_cv_robustness.main()

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["totals"]["records"] == 2
    assert report["totals"]["crashes"] == 1
    assert report["totals"]["schema_valid"] == 1
    assert report["kpis"]["crash_rate"] == 0.5
    assert report["kpis"]["schema_valid_rate"] == 0.5
