from datetime import date
from types import SimpleNamespace

import pytest

import app.services.matching as matching


class _FakeEmbeddingService:
    def generate_embedding(self, _text: str):
        return [1.0, 0.0, 0.0]


def test_semantic_similarity_normalizes_negative_cosine(monkeypatch):
    monkeypatch.setenv("ENABLE_AI_MATCHING", "true")
    monkeypatch.setattr(matching, "_get_embedding_service", lambda: _FakeEmbeddingService())
    monkeypatch.setattr(matching, "compute_semantic_similarity", lambda *_: -0.5)

    sim = matching._semantic_similarity("web developer html css", "content creator marketing")
    assert sim == pytest.approx(0.25, abs=1e-9)


def test_calculate_weighted_score_remains_non_negative_with_negative_cosine(monkeypatch):
    monkeypatch.setenv("ENABLE_AI_MATCHING", "true")
    monkeypatch.setattr(matching, "_get_embedding_service", lambda: _FakeEmbeddingService())
    monkeypatch.setattr(matching, "compute_semantic_similarity", lambda *_: -1.0)

    employee = SimpleNamespace(
        id=1,
        hire_date=date.today().replace(year=date.today().year - 2),
        performance_score="good",
        position="Content Creator",
        department="Marketing",
        skills=[SimpleNamespace(name="copywriting")],
    )

    score = matching.calculate_weighted_score(
        employee=employee,
        job_title="Web Developer",
        required_skills=["html", "css", "javascript"],
        min_experience=2,
        use_semantic=True,
    )

    assert score["semantic_similarity"] == 0.0
    assert score["total"] >= 0.0
