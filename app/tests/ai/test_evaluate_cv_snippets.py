import json
import sys
from pathlib import Path
import uuid

from app.scripts import evaluate_cv_snippets


def _local_tmp_dir() -> Path:
    root = Path("artifacts") / "pytest_local" / f"snippet_eval_{uuid.uuid4().hex[:10]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_evaluate_cv_snippets_generates_metrics_report(monkeypatch):
    root = _local_tmp_dir()
    gold = root / "gold.jsonl"
    gold.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "g1",
                        "text": "SKILLS\n- Python\nLANGUES\n- Francais (B1)",
                        "labels": {
                            "skills": ["python"],
                            "tools": [],
                            "languages": [{"language": "french", "level": "B1"}],
                            "title": None,
                            "experience_years": None,
                            "project_text": [],
                        },
                    }
                ),
                json.dumps(
                    {
                        "id": "g2",
                        "text": "implementation of a gps-based bus geolocation",
                        "labels": {
                            "skills": [],
                            "tools": [],
                            "languages": [],
                            "title": None,
                            "experience_years": None,
                            "project_text": ["implementation of a gps-based bus geolocation"],
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    out = root / "cv_eval_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_cv_snippets", "--gold-jsonl", str(gold), "--out", str(out)],
    )
    evaluate_cv_snippets.main()
    report = json.loads(out.read_text(encoding="utf-8"))

    assert report["config"]["records"] == 2
    assert "skills" in report["metrics"]
    assert "tools" in report["metrics"]
    assert "languages" in report["metrics"]
    assert "project_text_fp_rate" in report["metrics"]
