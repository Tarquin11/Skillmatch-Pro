import json
import sys
from pathlib import Path
import uuid

from app.scripts import evaluate_cv_quality_gates


def _local_tmp_dir() -> Path:
    root = Path("artifacts") / "pytest_local" / f"quality_gate_eval_{uuid.uuid4().hex[:10]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_evaluate_cv_quality_gates_passes_when_targets_met(monkeypatch):
    root = _local_tmp_dir()
    gold = root / "quality_gold.jsonl"
    gold.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "case_1",
                        "text": "CASE1",
                        "labels": {
                            "technical_skills": ["python", "sql"],
                            "certifications": ["aws cert"],
                            "hands_on_projects": ["api project"],
                            "known_skills": ["python", "sql", "docker"],
                        },
                    }
                ),
                json.dumps(
                    {
                        "id": "case_2",
                        "text": "CASE2",
                        "labels": {
                            "technical_skills": ["docker"],
                            "known_skills": ["docker"],
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    def _fake_parse(**kwargs):
        text = kwargs["file_bytes"].decode("utf-8", errors="ignore")
        if "CASE1" in text:
            return {
                "ok": True,
                "extracted_skills": [
                    {"skill": "python", "confidence": 0.99, "source": "exact:skills"},
                    {"skill": "sql", "confidence": 0.98, "source": "semantic_augment:skills"},
                ],
                "certifications": ["AWS Cert"],
                "hands_on_projects": ["API Project"],
            }
        return {
            "ok": True,
            "extracted_skills": [
                {"skill": "docker", "confidence": 0.97, "source": "exact:skills"},
            ],
            "certifications": [],
            "hands_on_projects": [],
        }

    monkeypatch.setattr(evaluate_cv_quality_gates, "parse_cv_safe", _fake_parse)

    out = root / "quality_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_cv_quality_gates",
            "--gold-jsonl",
            str(gold),
            "--out",
            str(out),
        ],
    )
    evaluate_cv_quality_gates.main()
    report = json.loads(out.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["metrics"]["technical_skills"]["precision_at_1"] == 1.0
    assert report["metrics"]["technical_skills"]["f1"] == 1.0
    assert report["metrics"]["semantic_only"]["semantic_augment_fp_rate"] == 0.0
    assert report["metrics"]["boards_quality"]["mutual_exclusivity_accuracy"] == 1.0
    assert report["gates"]["ece"]["passed"] is True


def test_evaluate_cv_quality_gates_fails_on_semantic_fp(monkeypatch):
    root = _local_tmp_dir()
    gold = root / "quality_gold_fail.jsonl"
    gold.write_text(
        json.dumps(
            {
                "id": "bad_case",
                "text": "BAD_CASE",
                "labels": {
                    "technical_skills": ["python"],
                    "known_skills": ["python", "go"],
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evaluate_cv_quality_gates,
        "parse_cv_safe",
        lambda **_kwargs: {
            "ok": True,
            "extracted_skills": [
                {"skill": "python", "confidence": 0.95, "source": "exact:skills"},
                {"skill": "go", "confidence": 0.90, "source": "semantic_augment:skills"},
            ],
            "certifications": [],
            "hands_on_projects": [],
        },
    )

    out = root / "quality_report_fail.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_cv_quality_gates",
            "--gold-jsonl",
            str(gold),
            "--out",
            str(out),
        ],
    )
    evaluate_cv_quality_gates.main()
    report = json.loads(out.read_text(encoding="utf-8"))

    assert report["passed"] is False
    assert report["gates"]["semantic_augment_fp_rate"]["passed"] is False
    assert report["gates"]["technical_f1"]["passed"] is False
