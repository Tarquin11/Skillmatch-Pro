import pytest
import numpy as np

from app.ai.feature_engineering import FeatureEngineer


def test_semantic_fallback_uses_lexical_signal_when_embedding_fails(monkeypatch):
    engineer = FeatureEngineer(use_semantic=True)
    monkeypatch.setattr(
        engineer,
        "_get_cached_embedding",
        lambda _text: (_ for _ in ()).throw(RuntimeError("embedding unavailable")),
    )

    bundle = engineer.create_features_with_debug(
        employee_raw={
            "position": "web developer",
            "department": "it",
            "skills": {"javascript"},
            "experience_years": 4.0,
            "performance_score": 0.5,
            "engagement_score": 0.5,
            "satisfaction_score": 0.5,
            "tenure_years": 4.0,
            "currently_active": 1.0,
        },
        job_raw={
            "title": "web developer",
            "required_skills": {"javascript", "html"},
            "required_experience_years": 2.0,
        },
    )

    assert bundle["features"]["semantic_similarity"] > 0.0
    assert bundle["debug"]["semantic_source"] == "lexical_fallback"


def test_related_skill_partial_credit_is_applied():
    engineer = FeatureEngineer(use_semantic=False)
    bundle = engineer.create_features_with_debug(
        employee_raw={
            "position": "frontend engineer",
            "department": "it",
            "skills": {"react"},
            "experience_years": 3.0,
            "performance_score": 0.5,
            "engagement_score": 0.5,
            "satisfaction_score": 0.5,
            "tenure_years": 3.0,
            "currently_active": 1.0,
        },
        job_raw={
            "title": "web developer",
            "required_skills": {"javascript"},
            "required_experience_years": 1.0,
        },
    )

    assert bundle["features"]["skill_overlap"] == 0.8
    assert bundle["debug"]["overlap_credits"]["javascript"] == 0.8


def test_mysql_sql_related_skill_gets_full_credit():
    engineer = FeatureEngineer(use_semantic=False)
    bundle = engineer.create_features_with_debug(
        employee_raw={
            "position": "backend engineer",
            "department": "it",
            "skills": {"sql"},
            "experience_years": 3.0,
            "performance_score": 0.5,
            "engagement_score": 0.5,
            "satisfaction_score": 0.5,
            "tenure_years": 3.0,
            "currently_active": 1.0,
        },
        job_raw={
            "title": "backend developer",
            "required_skills": {"mysql"},
            "required_experience_years": 1.0,
        },
    )

    assert bundle["features"]["skill_overlap"] == 1.0
    assert bundle["debug"]["overlap_credits"]["mysql"] == 1.0


def test_create_features_with_debug_contains_intermediates():
    engineer = FeatureEngineer(use_semantic=False)
    bundle = engineer.create_features_with_debug(
        employee_raw={
            "position": "php developer",
            "department": "it",
            "skills": {"js", "php oop"},
            "experience_years": 6.0,
            "performance_score": 0.7,
            "engagement_score": 0.5,
            "satisfaction_score": 0.5,
            "tenure_years": 6.0,
            "currently_active": 1.0,
        },
        job_raw={
            "title": "web developer",
            "required_skills": {"javascript", "php"},
            "required_experience_years": 2.0,
        },
    )

    debug = bundle["debug"]
    assert "job_text" in debug
    assert "employee_text" in debug
    assert "required_skills_canonical" in debug
    assert "owned_skills_canonical" in debug
    assert "semantic_source" in debug


def test_strict_semantic_fails_fast_when_embedding_unavailable(monkeypatch):
    monkeypatch.setenv("MATCH_STRICT_SEMANTIC", "true")
    monkeypatch.setattr(
        FeatureEngineer,
        "_get_embedding_service",
        lambda _self: (_ for _ in ()).throw(RuntimeError("embedding unavailable")),
    )
    with pytest.raises(RuntimeError, match="Semantic embeddings are required but unavailable"):
        FeatureEngineer(use_semantic=True)


def test_semantic_fallback_logs_warning_once(monkeypatch, caplog):
    monkeypatch.setenv("MATCH_STRICT_SEMANTIC", "false")
    engineer = FeatureEngineer(use_semantic=True)
    monkeypatch.setattr(
        engineer,
        "_get_cached_embedding",
        lambda _text: (_ for _ in ()).throw(RuntimeError("embedding unavailable")),
    )

    with caplog.at_level("WARNING"):
        engineer.create_features_with_debug(
            employee_raw={
                "position": "web developer",
                "department": "it",
                "skills": {"javascript"},
                "experience_years": 4.0,
                "performance_score": 0.5,
                "engagement_score": 0.5,
                "satisfaction_score": 0.5,
                "tenure_years": 4.0,
                "currently_active": 1.0,
            },
            job_raw={
                "title": "web developer",
                "required_skills": {"javascript", "html"},
                "required_experience_years": 2.0,
            },
        )
        engineer.create_features_with_debug(
            employee_raw={
                "position": "web developer",
                "department": "it",
                "skills": {"javascript"},
                "experience_years": 4.0,
                "performance_score": 0.5,
                "engagement_score": 0.5,
                "satisfaction_score": 0.5,
                "tenure_years": 4.0,
                "currently_active": 1.0,
            },
            job_raw={
                "title": "web developer",
                "required_skills": {"javascript", "html"},
                "required_experience_years": 2.0,
            },
        )

    lexical_warnings = [
        rec for rec in caplog.records if "Semantic similarity fallback triggered (lexical_fallback)" in rec.message
    ]
    assert len(lexical_warnings) == 1


def test_zero_embedding_similarity_is_not_treated_as_failure(monkeypatch):
    engineer = FeatureEngineer(use_semantic=True)
    monkeypatch.setattr(
        engineer,
        "_get_cached_embedding",
        lambda _text: np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )
    monkeypatch.setattr("app.ai.feature_engineering.compute_semantic_similarity", lambda *_: 0.0)

    bundle = engineer.create_features_with_debug(
        employee_raw={
            "position": "web developer",
            "department": "it",
            "skills": {"html"},
            "experience_years": 2.0,
            "performance_score": 0.5,
            "engagement_score": 0.5,
            "satisfaction_score": 0.5,
            "tenure_years": 2.0,
            "currently_active": 1.0,
        },
        job_raw={
            "title": "web developer",
            "required_skills": {"html", "css"},
            "required_experience_years": 1.0,
        },
    )

    assert bundle["debug"]["semantic_source"] == "embedding_blended"
    assert bundle["features"]["semantic_similarity"] > 0.0
