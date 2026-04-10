from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from app.ai.preprocessing import normalize_skill_name, preprocess_employee, preprocess_job
from app.services.embedding_service import compute_semantic_similarity

logger = logging.getLogger(__name__)


FEATURE_COLUMNS = [
    "skill_overlap",
    "missing_skill_ratio",
    "experience_gap",
    "experience_surplus",
    "semantic_similarity",
    "performance_score",
    "engagement_score",
    "satisfaction_score",
    "tenure_years",
    "currently_active",
]

SKILL_ALIASES = {
    "js": "javascript",
    "reactjs": "react",
    "react js": "react",
    "vuejs": "vue",
    "vue js": "vue",
    "angularjs": "angular",
    "nodejs": "node.js",
    "node js": "node.js",
    "html5": "html",
    "css3": "css",
    "mysql server": "mysql",
    "postgres": "postgresql",
    "postgresql server": "postgresql",
    "php oop": "php",
    "tailwindcss": "tailwind",
}

# Directional related-skill weights.
# key: required skill -> value: candidate skill -> overlap credit
SKILL_DISTANCE_TABLE = {
    "mysql": {"sql": 1.0, "mariadb": 0.8, "postgresql": 0.7},
    "sql": {"mysql": 1.0, "postgresql": 0.8, "mariadb": 0.8},
    "javascript": {
        "react": 0.8,
        "vue": 0.8,
        "angular": 0.8,
        "node.js": 0.75,
        "typescript": 0.75,
        "jquery": 0.7,
    },
    "python": {"fastapi": 0.9, "flask": 0.85, "django": 0.85},
    "php": {"laravel": 0.85, "symfony": 0.85, "zend": 0.8},
    "tailwind": {"css": 0.6},
    "css": {"tailwind": 0.7},
}

POSITION_SKILL_HINTS = {
    "frontend developer": {"html", "css", "javascript"},
    "front end developer": {"html", "css", "javascript"},
    "web developer": {"html", "css", "javascript"},
    "full stack developer": {"html", "css", "javascript", "sql"},
    "backend developer": {"sql"},
    "back end developer": {"sql"},
    "php developer": {"php", "html", "css", "javascript"},
}

TEXT_STOPWORDS = {
    "and",
    "or",
    "the",
    "a",
    "an",
    "with",
    "of",
    "for",
    "in",
    "to",
    "on",
    "at",
    "by",
    "from",
}


class FeatureEngineer:
    def __init__(self, use_semantic: bool = True):
        self.use_semantic = use_semantic
        self.strict_semantic = self._env_flag("MATCH_STRICT_SEMANTIC", default=False)
        self._embedding_service = None
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._warned_semantic_sources: set[str] = set()
        if self.use_semantic and self.strict_semantic:
            try:
                self._get_embedding_service()
            except Exception as exc:
                raise RuntimeError(
                    "Semantic embeddings are required but unavailable. "
                    "Install embedding dependencies or set MATCH_STRICT_SEMANTIC=false."
                ) from exc

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        raw = str(os.getenv(name, "")).strip().lower()
        if not raw:
            return default
        return raw in {"1", "true", "yes", "on"}

    def _warn_semantic_fallback_once(self, source: str, detail: str) -> None:
        if source in self._warned_semantic_sources:
            return
        self._warned_semantic_sources.add(source)
        logger.warning(
            "Semantic similarity fallback triggered (%s): %s",
            source,
            detail,
        )

    @staticmethod
    def _normalize_embedding_similarity(raw_similarity: float) -> float:
        if not np.isfinite(raw_similarity):
            return 0.0
        clipped = float(max(-1.0, min(1.0, raw_similarity)))
        return float((clipped + 1.0) / 2.0)

    def _get_embedding_service(self):
        if self._embedding_service is None:
            from app.services.embedding_service import EmbeddingService

            self._embedding_service = EmbeddingService()
        return self._embedding_service

    def _canonical_skill(self, skill: Any) -> str:
        normalized = normalize_skill_name(skill)
        if not normalized:
            return ""
        return SKILL_ALIASES.get(normalized, normalized)

    def _canonicalize_skills(self, skills: set[str]) -> set[str]:
        out: set[str] = set()
        for skill in skills or set():
            canonical = self._canonical_skill(skill)
            if canonical:
                out.add(canonical)
        return out

    def _infer_skills_from_position(self, position: str) -> set[str]:
        normalized_position = normalize_skill_name(position)
        if not normalized_position:
            return set()

        inferred: set[str] = set()
        for marker, hints in POSITION_SKILL_HINTS.items():
            if marker in normalized_position:
                inferred.update(hints)
        return self._canonicalize_skills(inferred)

    def _job_text(self, job: dict[str, Any]) -> str:
        return " ".join(
            [
                str(job.get("title", "") or ""),
                str(job.get("description", "") or ""),
                " ".join(sorted(job.get("required_skills", set()))),
            ]
        ).strip()

    def _employee_text(self, employee: dict[str, Any]) -> str:
        inferred = self._infer_skills_from_position(str(employee.get("position", "") or ""))
        owned = self._canonicalize_skills(set(employee.get("skills", set()) or set()))
        combined = sorted(owned | inferred)
        return " ".join(
            [
                str(employee.get("position", "") or ""),
                str(employee.get("department", "") or ""),
                " ".join(combined),
            ]
        ).strip()

    def _tokenize_for_similarity(self, text: str) -> set[str]:
        tokens = [normalize_skill_name(tok) for tok in str(text or "").split()]
        return {tok for tok in tokens if tok and tok not in TEXT_STOPWORDS and len(tok) > 1}

    def _lexical_similarity(self, job_text: str, employee_text: str) -> float:
        job_tokens = self._tokenize_for_similarity(job_text)
        employee_tokens = self._tokenize_for_similarity(employee_text)
        if not job_tokens or not employee_tokens:
            return 0.0

        overlap = len(job_tokens & employee_tokens)
        union = len(job_tokens | employee_tokens)
        cosine_like = overlap / float(np.sqrt(len(job_tokens) * len(employee_tokens)))
        smoothed = (overlap + 1.0) / (union + 8.0)
        return float(max(0.0, min(1.0, (0.7 * cosine_like) + (0.3 * smoothed))))

    def _semantic_similarity_with_source(self, employee: dict[str, Any], job: dict[str, Any]) -> tuple[float, str]:
        if not self.use_semantic:
            return 0.0, "disabled"

        job_text = self._job_text(job)
        employee_text = self._employee_text(employee)
        if not job_text or not employee_text:
            return 0.0, "empty_text"

        lexical = self._lexical_similarity(job_text, employee_text)

        try:
            v_job = self._get_cached_embedding(job_text)
            v_emp = self._get_cached_embedding(employee_text)

            if v_job.size == 0 or v_emp.size == 0 or v_job.shape != v_emp.shape:
                self._warn_semantic_fallback_once(
                    "embedding_invalid_lexical_fallback",
                    "Embedding vectors were empty or incompatible; using lexical fallback.",
                )
                return lexical, "embedding_invalid_lexical_fallback"

            raw_similarity = float(compute_semantic_similarity(v_job, v_emp))
            if not np.isfinite(raw_similarity):
                self._warn_semantic_fallback_once(
                    "embedding_non_finite_lexical_fallback",
                    "Embedding similarity was non-finite; using lexical fallback.",
                )
                return lexical, "embedding_non_finite_lexical_fallback"

            # Dot product over normalized embeddings is cosine in [-1, 1];
            # map to [0, 1] so scoring weights remain meaningful.
            embedding_similarity = self._normalize_embedding_similarity(raw_similarity)
            blended = (0.85 * embedding_similarity) + (0.15 * lexical)
            return float(max(0.0, min(1.0, blended))), "embedding_blended"
        except Exception as exc:
            if self.strict_semantic:
                raise RuntimeError(
                    "Semantic similarity failed while MATCH_STRICT_SEMANTIC=true."
                ) from exc
            self._warn_semantic_fallback_once(
                "lexical_fallback",
                f"Embedding path failed with {type(exc).__name__}; using lexical fallback.",
            )
            return lexical, "lexical_fallback"

    def _skill_overlap_with_debug(self, required: set[str], owned: set[str]) -> tuple[float, dict[str, float]]:
        if not required:
            return 0.0, {}

        credits: dict[str, float] = {}
        total_credit = 0.0
        for req in sorted(required):
            credit = 0.0
            if req in owned:
                credit = 1.0
            else:
                for own in owned:
                    configured_credit = SKILL_DISTANCE_TABLE.get(req, {}).get(own)
                    if configured_credit is not None:
                        credit = max(credit, float(configured_credit))
                if credit <= 0.0 and any((req in own or own in req) for own in owned):
                    credit = 0.4
            credits[req] = float(credit)
            total_credit += credit

        overlap = total_credit / float(len(required))
        return float(max(0.0, min(1.0, overlap))), credits

    def _build_features_and_debug(self, employee: dict[str, Any], job: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
        required_raw = set(job.get("required_skills", set()) or set())
        owned_raw = set(employee.get("skills", set()) or set())
        inferred_skills = self._infer_skills_from_position(str(employee.get("position", "") or ""))

        required = self._canonicalize_skills(required_raw)
        owned = self._canonicalize_skills(owned_raw) | inferred_skills

        skill_overlap, overlap_credits = self._skill_overlap_with_debug(required, owned)
        missing_skill_ratio = 1.0 - skill_overlap if required else 0.0

        req_exp = float(job.get("required_experience_years", 0.0))
        emp_exp = float(employee.get("experience_years", 0.0))
        experience_gap = max(req_exp - emp_exp, 0.0)
        experience_surplus = max(emp_exp - req_exp, 0.0)

        semantic_similarity, semantic_source = self._semantic_similarity_with_source(employee, job)

        features = {
            "skill_overlap": float(skill_overlap),
            "missing_skill_ratio": float(missing_skill_ratio),
            "experience_gap": float(experience_gap),
            "experience_surplus": float(experience_surplus),
            "semantic_similarity": float(semantic_similarity),
            "performance_score": float(employee.get("performance_score", 0.0)),
            "engagement_score": float(employee.get("engagement_score", 0.0)),
            "satisfaction_score": float(employee.get("satisfaction_score", 0.0)),
            "tenure_years": float(employee.get("tenure_years", 0.0)),
            "currently_active": float(employee.get("currently_active", 0.0)),
        }

        debug = {
            "job_text": self._job_text(job),
            "employee_text": self._employee_text(employee),
            "required_skills_raw": sorted(required_raw),
            "required_skills_canonical": sorted(required),
            "owned_skills_raw": sorted(owned_raw),
            "inferred_skills": sorted(inferred_skills),
            "owned_skills_canonical": sorted(owned),
            "overlap_credits": overlap_credits,
            "semantic_source": semantic_source,
            "semantic_similarity": float(semantic_similarity),
        }
        return features, debug

    def create_features(self, employee_raw: Any, job_raw: Any) -> dict[str, float]:
        bundle = self.create_features_with_debug(employee_raw, job_raw)
        return bundle["features"]

    def create_features_with_debug(self, employee_raw: Any, job_raw: Any) -> dict[str, Any]:
        employee = (
            employee_raw
            if isinstance(employee_raw, dict) and "skills" in employee_raw
            else preprocess_employee(employee_raw)
        )
        job = (
            job_raw
            if isinstance(job_raw, dict) and "required_skills" in job_raw
            else preprocess_job(job_raw)
        )
        features, debug = self._build_features_and_debug(employee, job)
        return {"features": features, "debug": debug}

    def vectorize_pairs(self, pairs: list[dict[str, Any]]):
        if self.use_semantic:
            self.precompute_embeddings(pairs, batch_size=256)
        rows = []
        labels = []
        for row in pairs:
            feats = self.create_features(row["employee"], row["job"])
            rows.append([feats[col] for col in FEATURE_COLUMNS])
            if "label" in row:
                labels.append(int(row["label"]))

        x = np.asarray(rows, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int32) if labels else None
        return x, y

    def _get_cached_embedding(self, text: str) -> np.ndarray:
        if not text:
            return np.array([], dtype=np.float32)

        key = text.strip().lower()
        if not key:
            return np.array([], dtype=np.float32)

        if key in self._embedding_cache:
            return self._embedding_cache[key]

        svc = self._get_embedding_service()
        vec = np.asarray(svc.generate_embedding(key), dtype=np.float32)
        self._embedding_cache[key] = vec
        return vec

    def precompute_embeddings(self, pairs: list[dict[str, Any]], batch_size: int = 256) -> None:
        if not self.use_semantic:
            return

        unique_texts = set()
        for row in pairs:
            employee = (
                row["employee"]
                if isinstance(row["employee"], dict) and "skills" in row["employee"]
                else preprocess_employee(row["employee"])
            )
            job = (
                row["job"]
                if isinstance(row["job"], dict) and "required_skills" in row["job"]
                else preprocess_job(row["job"])
            )

            jt = self._job_text(job).strip().lower()
            et = self._employee_text(employee).strip().lower()
            if jt:
                unique_texts.add(jt)
            if et:
                unique_texts.add(et)

        texts = [t for t in unique_texts if t and t not in self._embedding_cache]
        if not texts:
            return

        svc = self._get_embedding_service()
        vectors = svc.generate_embeddings(texts, batch_size=batch_size)
        for text, vec in zip(texts, vectors):
            self._embedding_cache[text] = np.asarray(vec, dtype=np.float32)
