from __future__ import annotations
from typing import Any, Sequence
import numpy as np
from app.ai.feature_engineering import FEATURE_COLUMNS
from app.ai.runtime import get_matcher
from app.services.matching import calculate_weighted_score

class ModelInferenceService:
    def rank_candidates(self,*,job_title: str,required_skills: Sequence[str],min_experience: int, employees: Sequence[Any],limit: int, ) -> list[dict[str, Any]]:
        matcher = get_matcher()
        if matcher and getattr(matcher, "is_fitted", False):
            ranked = self._rank_with_model(matcher=matcher, job_title=job_title, required_skills=required_skills, min_experience=min_experience, employees=employees,)
        else: 
            ranked = self._rank_with_heuristic(
                job_title=job_title,
                required_skills=required_skills, 
                min_experience=min_experience, 
                employees=employees,
                )

        ranked.sort(key=lambda x: x["predicted_fit_score"], reverse=True)
        return ranked[:limit]
    
    def _rank_with_model(
        self, 
        *, 
        matcher: Any, 
        job_title: str, 
        required_skills: Sequence[str], 
        min_experience: int, 
        employees: Sequence[Any]
    ) -> list[dict[str, Any]]:
        job_payload = {
            "title": job_title,
            "required_skills": list(required_skills),
            "min_experience": min_experience,
        }

        out: list[dict[str, Any]] = []
        
        for employee in employees:
            pred = matcher.predict_score(employee, job_payload)
            features = pred.get("features", {}) or {}
            breakdown = self._model_feature_breakdown(matcher, features)
            reasons = self._top_reasons(features, breakdown)
            score = float(pred.get("score_percent", 0.0))
            out.append({
                "employee_id": int(employee.id),
                "full_name":self._full_name(employee),
                "score": round(score, 2),
                "predicted_fit_score": round(score, 2),
                "scoring_source": "model",
                "feature_breakdown": breakdown,
                "top_reasons": reasons,
            })
            
        return out 
    
    def _rank_with_heuristic(
        self,
        *,
        job_title: str,
        required_skills: Sequence[str],
        min_experience: int,
        employees: Sequence[Any],
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for employee in employees:
            h = calculate_weighted_score(
                employee=employee,
                job_title=job_title,
                required_skills=required_skills,
                min_experience=min_experience,
            )
            score = float(h.get("total", 0.0))
            breakdown = {
                "skill_overlap": float(h.get("skill_overlap", 0.0)),
                "experience_score": float(h.get("experience_score", 0.0)),
                "semantic_similarity": float(h.get("semantic_similarity", 0.0)),
                "performance_score": float(h.get("performance_score", 0.0)),
            }
            reasons = self._top_reasons(breakdown, breakdown)

            out.append(
                {
                    "employee_id": int(employee.id),
                    "full_name": self._full_name(employee),
                    "score": round(score, 2),
                    "predicted_fit_score": round(score, 2),
                    "scoring_source": "heuristic",
                    "feature_breakdown": {k: round(v, 4) for k, v in breakdown.items()},
                    "top_reasons": reasons,
                }
            )
        return out

    def _model_feature_breakdown(self, matcher: Any, features: dict[str, float]) -> dict[str, float]:
        vector = np.asarray([float(features.get(c, 0.0)) for c in FEATURE_COLUMNS], dtype=np.float32)
        pipeline = getattr(matcher, "model", None)
        if pipeline is None:
            return {c: 0.0 for c in FEATURE_COLUMNS}

        named_steps = getattr(pipeline, "named_steps", {})
        scaler = named_steps.get("scaler")
        clf = named_steps.get("clf", pipeline)

        if hasattr(clf, "coef_"):
            x = vector.reshape(1, -1)
            if scaler is not None:
                x = scaler.transform(x)
            contrib = x[0] * clf.coef_[0]
            return {c: round(float(v), 6) for c, v in zip(FEATURE_COLUMNS, contrib)}

        if hasattr(clf, "feature_importances_"):
            fi = np.asarray(clf.feature_importances_, dtype=np.float32)
            contrib = vector * fi
            return {c: round(float(v), 6) for c, v in zip(FEATURE_COLUMNS, contrib)}

        return {c: round(float(features.get(c, 0.0)), 6) for c in FEATURE_COLUMNS}

    def _top_reasons(self, features: dict[str, float], breakdown: dict[str, float], top_n: int = 3) -> list[str]:
        items = list(breakdown.items())
        positives = [(k, v) for k, v in items if v > 0]
        ordered = sorted(positives, key=lambda x: x[1], reverse=True) or sorted(
            items, key=lambda x: abs(x[1]), reverse=True
        )
        return [self._reason_text(name, float(features.get(name, 0.0)), float(val)) for name, val in ordered[:top_n]]

    def _reason_text(self, feature: str, value: float, contribution: float) -> str:
        if feature == "skill_overlap":
            return f"Strong skill Overlap ({value:.2f})."
        if feature == "experience_surplus":
            return f"Experience exceeds requirement ({value:.2f} years)."
        if feature == "experience_gap":
            return f"Low experience gap ({value:.2f} years)."
        if feature == "semantic_similarity":
            return f"High semantic similarity ({value:.2f})."
        if feature == "performance_score":
            return f"Good performance history ({value:.2f})."
        if feature == "currently_active":
            return "Currently active employee profile."
        direction = "positive" if contribution >= 0 else "negative"
        return f"{feature.replace('_', ' ')} had {direction} impact ({contribution:+.3f})."

    def _full_name(self, employee: Any) -> str:
        name = (getattr(employee, "full_name", None) or "").strip()
        if name:
            return name
        first = (getattr(employee, "first_name", None) or "").strip()
        last = (getattr(employee, "last_name", None) or "").strip()
        return (f"{first} {last}".strip() or "Unknown")
