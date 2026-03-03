from __future__ import annotations
from typing import Any
import numpy as np
from app.ai.preprocessing import preprocess_employee, preprocess_job
from app.services.embedding_service import compute_semantic_similarity


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

class FeatureEngineer:
    def __init__(self, use_semantic: bool = True):
        self.use_semantic = use_semantic
        self._embedding_service = None
        self._embedding_cache: dict[str, np.ndarray] = {}

    def _get_embedding_service(self):
        if self._embedding_service is None:
            from app.services.embedding_service import EmbeddingService
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    def _job_text(self, job: dict[str, Any]) -> str:
        return " ".join(
            [
                job.get("title", ""),
                job.get("description", ""),
                " ".join(sorted(job.get("required_skills", set()))),
            ]
        ).strip()

    def _employee_text(self, employee: dict[str, Any]) -> str:
        return " ".join(
            [
                employee.get("position", ""),
                employee.get("department", ""),
                " ".join(sorted(employee.get("skills", set()))),
            ]
        ).strip()

    def _semantic_similarity(self, employee: dict[str, Any], job: dict[str, Any]) -> float:
        if not self.use_semantic:
            return 0.0

        job_text = self._job_text(job)
        employee_text = self._employee_text(employee)
        if not job_text or not employee_text:
            return 0.0

        try:
            v_job = self._get_cached_embedding(job_text)
            v_emp = self._get_cached_embedding(employee_text)
            return float(compute_semantic_similarity(v_job, v_emp))
        except Exception:
            return 0.0

    def create_features(self, employee_raw: Any, job_raw: Any) -> dict[str, float]:
        employee = employee_raw if isinstance(employee_raw, dict) and "skills" in employee_raw else preprocess_employee(employee_raw)
        job = job_raw if isinstance(job_raw, dict) and "required_skills" in job_raw else preprocess_job(job_raw)

        required = job["required_skills"]
        owned = employee["skills"]

        overlap_count = len(required & owned)
        required_count = len(required)

        skill_overlap = (overlap_count / required_count) if required_count > 0 else 0.0
        missing_skill_ratio = 1.0 - skill_overlap if required_count > 0 else 0.0

        req_exp = float(job.get("required_experience_years", 0.0))
        emp_exp = float(employee.get("experience_years", 0.0))
        experience_gap = max(req_exp - emp_exp, 0.0)
        experience_surplus = max(emp_exp - req_exp, 0.0)

        semantic_similarity = self._semantic_similarity(employee, job)

        return {
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
        self._embedding_cache[key]= vec
        return vec
    
    def precompute_embeddings(self, pairs: list[dict[str, Any]], batch_size: int = 256) -> None:
        if not self.use_semantic:
            return
        
        unique_texts = set()
        for row in pairs:
            employee = row["employee"] if isinstance(row["employee"], dict) and "skills" in row["employee"] else preprocess_employee(row["employee"])
            job = row["job"] if isinstance(row["job"], dict) and "required_skills" in row["job"] else preprocess_job(row["job"])

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
        for t,v in zip(texts, vectors):
            self._embedding_cache[t] = np.asarray(v, dtype=np.float32)
