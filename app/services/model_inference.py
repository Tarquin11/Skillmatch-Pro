from __future__ import annotations
import csv
import json
import logging
import os
import hashlib
import re
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, time
from typing import Any, Sequence
import numpy as np
import pandas as pd
from app.ai.feature_engineering import FEATURE_COLUMNS
from app.ai.preprocessing import normalize_skill_name
from app.ai.runtime import get_matcher
from app.core.structured_log import EVENT_AI_FALLBACK_USED,EVENT_AI_PREDICTION_FAILURE,REASON_FALLBACK_USED,REASON_MODEL_FAIL,log_structured_event
from app.services.matching import calculate_weighted_score
from app.services.training_recommendation import build_training_recommendations
from app.ai.matcher import CandidateMatcher

logger = logging.getLogger(__name__)

class ModelInferenceService:
    _SCORING_POLICY_VERSION = "2026-04-10-exp-filter-v1"
    _CACHE_MAX = 2048
    _CACHE_TTL_SEC = 300.0
    _PRED_CACHE: "OrderedDict[str, tuple[float, dict[str, Any]]]" = OrderedDict()

    _DRIFT_MONITORING_ENABLED = os.getenv(
        "AI_DRIFT_MONITORING_ENABLED",
        os.getenv("AI_EVIDENTLY_ENABLED", "true"),
    ).strip().lower() in {"1", "true", "yes", "on"}

    _DRIFT_MIN_ROWS = int(
        os.getenv(
            "AI_DRIFT_MIN_ROWS",
            os.getenv("AI_EVIDENTLY_MIN_ROWS", "30"),
        )
    )

    _DRIFT_BASELINE_PATH = Path(
        os.getenv(
            "AI_DRIFT_BASELINE_PATH",
            os.getenv("AI_EVIDENTLY_BASELINE_PATH", "artifacts/monitoring/reference_predictions.csv"),
        )
    )

    _WHYLOGS_REPORTS_DIR = Path(
        os.getenv("AI_WHYLOGS_REPORTS_DIR", "artifacts/monitoring/whylogs")
    )
    _MONITORING_EVENTS_PATH = Path(
        os.getenv("AI_MONITORING_EVENTS_PATH", "artifacts/monitoring/current_predictions.csv")
    )
    _DRIFT_EVENTS_PATH = Path(
        os.getenv("AI_DRIFT_EVENTS_PATH", "artifacts/monitoring/drift_metrics.jsonl")
    )

    _CALIBRATION_BINS = int(os.getenv("AI_CALIBRATION_BINS", "10"))
    _CALIBRATION_WARNING_MEAN_GT = float(os.getenv("AI_CALIBRATION_WARNING_MEAN_GT", "0.95"))
    _CALIBRATION_WARNING_STD_LT = float(os.getenv("AI_CALIBRATION_WARNING_STD_LT", "0.03"))
    _CANARY_ENABLED = os.getenv("AI_CANARY_ENABLED", "false").strip().lower() in {"1", "true","yes","on"}
    _CANARY_MODEL_PATH = Path(os.getenv("AI_CANARY_MODEL_PATH", "artifacts/matcher_canary.joblib"))

    try:
        _CANARY_TRAFFIC_PERCENT = float(os.getenv("AI_CANARY_TRAFFIC_PERCENT", "0"))
    except ValueError:
        _CANARY_TRAFFIC_PERCENT = 0.0
    _CANARY_TRAFFIC_PERCENT = min(100.0, max(0.0, _CANARY_TRAFFIC_PERCENT))
    _CANARY_STICKY_SALT = os.getenv("AI_CANARY_STICKY_SALT","skillmatch-canary-v1")
    _CANARY_MATCHER: Any | None = None
    _CANARY_LOAD_ATTEMTED = False
    _COLLAPSED_SCORE_MIN_ROWS = int(os.getenv("AI_COLLAPSED_SCORE_MIN_ROWS", "10"))
    _COLLAPSED_SCORE_MAX_UNIQUE = int(os.getenv("AI_COLLAPSED_SCORE_MAX_UNIQUE", "2"))
    _COLLAPSED_SCORE_STDDEV_MAX = float(os.getenv("AI_COLLAPSED_SCORE_STDDEV_MAX", "0.1"))
    _PROJECT_BONUS_MAX = float(os.getenv("AI_PROJECT_BONUS_MAX", "8.0"))

    @staticmethod
    def _bounded_unit(value: Any, default: float = 0.0) -> float:
        try:
            val = float(value)
        except (TypeError, ValueError):
            val = default
        return float(max(0.0, min(1.0, val)))

    def _apply_required_skill_guardrail(
        self,
        *,
        score_percent: float,
        skill_overlap: float,
        required_skills: Sequence[str],
    ) -> tuple[float, str | None]:
        if len(required_skills or []) == 0:
            return float(score_percent), None

        overlap = self._bounded_unit(skill_overlap, default=0.0)
        if overlap >= 0.35:
            return float(score_percent), None

        if overlap < 0.05:
            cap = 15.0
        elif overlap < 0.20:
            cap = 27.5
        else:
            cap = 40.0

        adjusted = float(min(float(score_percent), cap))
        if adjusted >= float(score_percent):
            return float(score_percent), None

        reason = (
            f"Low required-skill coverage ({overlap:.2f}) limited score to {cap:.1f}%."
        )
        return adjusted, reason

    def _apply_experience_guardrail(
        self,
        *,
        score_percent: float,
        predicted_experience_years: float,
        min_experience: int,
    ) -> tuple[float, str | None]:
        if int(min_experience or 0) <= 0:
            return float(score_percent), None

        exp = max(0.0, float(predicted_experience_years or 0.0))
        required = float(max(1, int(min_experience)))
        if exp >= required:
            return float(score_percent), None

        ratio = exp / required
        if ratio < 0.25:
            cap = 10.0
        elif ratio < 0.50:
            cap = 14.0
        elif ratio < 0.75:
            cap = 18.0
        else:
            cap = 24.0

        adjusted = float(min(float(score_percent), cap))
        if adjusted >= float(score_percent):
            return float(score_percent), None

        reason = (
            f"Experience below requirement ({exp:.2f}y / {required:.2f}y) limited score to {cap:.1f}%."
        )
        return adjusted, reason

    def _title_alignment_ratio(self, job_title: str, predicted_title: str | None) -> float:
        job_tokens = {tok for tok in normalize_skill_name(job_title).split() if tok}
        title_tokens = {tok for tok in normalize_skill_name(predicted_title or "").split() if tok}
        if not job_tokens or not title_tokens:
            return 0.0
        return float(len(job_tokens & title_tokens) / len(job_tokens))

    def _is_relevant_row(
        self,
        *,
        row: dict[str, Any],
        job_title: str,
        required_skills: Sequence[str],
        min_experience: int,
    ) -> bool:
        if len(required_skills or []) == 0:
            return True

        required_exp = int(min_experience or 0)
        if required_exp > 0:
            try:
                predicted_exp = float(row.get("predicted_experience_years", 0.0))
            except (TypeError, ValueError):
                predicted_exp = 0.0
            if predicted_exp < float(required_exp):
                # Keep only strongly skill-aligned exceptions below experience threshold.
                try:
                    overlap = 1.0 - float(row.get("skill_gap_ratio", 1.0))
                except (TypeError, ValueError):
                    overlap = 0.0
                if overlap < 0.5:
                    return False

        matched = row.get("matched_skills")
        if isinstance(matched, (list, tuple, set)) and len(matched) > 0:
            return True

        try:
            gap_ratio = float(row.get("skill_gap_ratio", 1.0))
            if gap_ratio < 0.999:
                return True
        except (TypeError, ValueError):
            pass

        title_align = self._title_alignment_ratio(
            job_title,
            str(row.get("predicted_title") or ""),
        )
        return title_align >= 0.5

    def _filter_low_relevance_rows(
        self,
        *,
        rows: Sequence[dict[str, Any]],
        job_title: str,
        required_skills: Sequence[str],
        min_experience: int,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        if len(required_skills or []) == 0:
            return list(rows)

        relevant = [
            row
            for row in rows
            if self._is_relevant_row(
                row=row,
                job_title=job_title,
                required_skills=required_skills,
                min_experience=min_experience,
            )
        ]
        if not relevant:
            return list(rows)

        dropped = len(rows) - len(relevant)
        if dropped > 0:
            logger.info(
                "ai_relevance_filter_applied job_title=%s total=%d kept=%d dropped=%d",
                job_title,
                len(rows),
                len(relevant),
                dropped,
            )
        return relevant

    def _log_inference_metrics(
        self,
        *,
        source: str,
        job_title: str,
        total: int,
        success: int,
        failed: int,
        latency_ms: float,
        batch: bool,
    ) -> None:
        logger.info(
            "ai_inference_metrics source=%s job_title=%s batch=%s total=%d success=%d failed=%d latency_ms=%.2f",
            source,
            job_title,
            batch,
            total,
            success,
            failed,
            latency_ms,
        )

    def _log_structured_failure(
        self,
        *,
        reason: str,
        stage: str,
        source: str,
        job_title: str,
        employee_id: int | None = None,
    ) -> None:
        log_structured_event(
            logger,
            level=logging.ERROR,
            event=EVENT_AI_PREDICTION_FAILURE,
            reason=reason,
            stage=stage,
            source=source,
            job_title=job_title,
            employee_id=employee_id,
        )

    def _log_structured_fallback(
        self,
        *,
        stage: str,
        fallback_source: str,
        job_title: str,
        employee_id: int | None = None,
        details: str | None = None,
    ) -> None:
        log_structured_event(
            logger,
            level=logging.WARNING,
            event=EVENT_AI_FALLBACK_USED,
            reason=REASON_FALLBACK_USED,
            stage=stage,
            fallback_source=fallback_source,
            job_title=job_title,
            employee_id=employee_id,
            details=details,
        )

    def _job_cache_key(self, job_title: str, required_skills: Sequence[str], min_experience: int) -> str:
        skills = [normalize_skill_name(s) for s in (required_skills or []) if s]
        skills_key = ",".join(sorted({s for s in skills if s}))
        return f"{self._SCORING_POLICY_VERSION}|{(job_title or '').strip().lower()}|{min_experience}|{skills_key}"

    def _employee_cache_key(self, employee: Any) -> str:
        emp_id = getattr(employee, "id", None)
        pos = (getattr(employee, "position", "") or "").strip().lower()
        exp = getattr(employee, "experience_years", "")
        skills = [normalize_skill_name(s) for s in self._extract_employee_skills(employee)]
        skills_key = ",".join(sorted({s for s in skills if s}))
        return f"id={emp_id}|pos={pos}|exp={exp}|skills={skills_key}"

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        item = self._PRED_CACHE.get(key)
        if not item:
            return None
        ts, payload = item
        if time() - ts > self._CACHE_TTL_SEC:
            self._PRED_CACHE.pop(key, None)
            return None
        self._PRED_CACHE.move_to_end(key)
        return payload

    def _cache_set(self, key: str, payload: dict[str, Any]) -> None:
        self._PRED_CACHE[key] = (time(), payload)
        self._PRED_CACHE.move_to_end(key)
        while len(self._PRED_CACHE) > self._CACHE_MAX:
            self._PRED_CACHE.popitem(last=False)
    
    @classmethod
    def _get_canary_matcher(cls) -> Any | None:
        if not cls._CANARY_ENABLED or cls._CANARY_TRAFFIC_PERCENT <= 0:
            return None
        if cls._CANARY_LOAD_ATTEMTED:
            return cls._CANARY_MATCHER
        
        cls._CANARY_LOAD_ATTEMTED = True
        if not cls._CANARY_MODEL_PATH.exists():
            logger.warning ("ai_canary_model_missing path=%s", cls._CANARY_MODEL_PATH)
            return None
        
        try:
            matcher = CandidateMatcher.load(cls._CANARY_MODEL_PATH)
            if not getattr(matcher, "is_fitted", False):
                logger.warning("ai_canary_model_unfitted path=%s", cls._CANARY_MODEL_PATH)
                return None
            cls._CANARY_MATCHER = matcher
            logger.info("ai_canary_model_loaded path=%s", cls._CANARY_MODEL_PATH)
            return cls._CANARY_MATCHER
        except Exception:
            logger.exception("ai_canary_model_load_failure path=%s", cls._CANARY_MODEL_PATH)
            return None
        
    def _stable_rollout_bucket(self, routing_key: str) -> float:
        digest = hashlib.sha256(f"{self._CANARY_STICKY_SALT}|{routing_key}".encode("utf-8")).hexdigest()
        return (int(digest[:8], 16) / 0xFFFFFFFF) * 100.0

    def _select_model_matcher(
        self,
        *,
        job_title: str,
        required_skills: Sequence[str],
        min_experience: int,
        routing_key: str | None,
    ) -> tuple[Any | None, str]:
        primary = get_matcher()
        if primary is None or not getattr(primary, "is_fitted", False):
            return None, "heuristic"

        canary = self._get_canary_matcher()
        if canary is None:
            return primary, "model_primary"

        seed = routing_key or self._job_cache_key(job_title, required_skills, min_experience)
        bucket = self._stable_rollout_bucket(seed)
        use_canary = bucket < self._CANARY_TRAFFIC_PERCENT
        source = "model_canary" if use_canary else "model_primary"

        logger.info(
            "ai_canary_routing source=%s bucket=%.4f threshold=%.2f key_hash=%s",
            source,
            bucket,
            self._CANARY_TRAFFIC_PERCENT,
            hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12],
        )
        return (canary if use_canary else primary), source

    def rank_candidates(
        self,
        *,
        job_title: str,
        required_skills: Sequence[str],
        min_experience: int,
        employees: Sequence[Any],
        limit: int,
        routing_key: str | None = None,
    ) -> list[dict[str, Any]]:
        matcher, model_source = self._select_model_matcher(
            job_title=job_title,
            required_skills=required_skills,
            min_experience=min_experience,
            routing_key=routing_key,
        )

        if matcher is not None:
            try:
                ranked = self._rank_with_model(
                    matcher=matcher,
                    job_title=job_title,
                    required_skills=required_skills,
                    min_experience=min_experience,
                    employees=employees,
                    model_source=model_source,
                )
            except Exception:
                self._log_structured_failure(
                    reason=REASON_MODEL_FAIL,
                    stage="rank_candidates_batch",
                    source=model_source,
                    job_title=job_title,
                )
                logger.exception(
                    "ai_prediction_failure scope=batch source=%s job_title=%s",
                    model_source,
                    job_title,
                )
                self._log_structured_fallback(
                    stage="rank_candidates_batch",
                    fallback_source="heuristic",
                    job_title=job_title,
                    details=f"{model_source}_batch_failed",
                )
                ranked = self._rank_with_heuristic(
                    job_title=job_title,
                    required_skills=required_skills,
                    min_experience=min_experience,
                    employees=employees,
                )
        else:
            self._log_structured_fallback(
                stage="rank_candidates_entry",
                fallback_source="heuristic",
                job_title=job_title,
                details="primary_model_unavailable",
            )
            ranked = self._rank_with_heuristic(
                job_title=job_title,
                required_skills=required_skills,
                min_experience=min_experience,
                employees=employees,
            )
        ranked.sort(key=lambda x: x["predicted_fit_score"], reverse=True)
        if matcher is not None and self._scores_look_collapsed(ranked):
            logger.warning(
                "ai_collapsed_scores_detected job_title=%s total=%d unique=%d std=%.4f source=%s",
                job_title,
                len(ranked),
                len({round(float(row.get("predicted_fit_score", 0.0)), 2) for row in ranked}),
                float(np.std([float(row.get("predicted_fit_score", 0.0)) for row in ranked])) if ranked else 0.0,
                model_source,
            )
            recovered = self._recover_collapsed_scores(
                rows=ranked,
                employees=employees,
                job_title=job_title,
                required_skills=required_skills,
                min_experience=min_experience,
                model_source=model_source,
            )
            if recovered:
                ranked = recovered
                ranked.sort(key=lambda x: x["predicted_fit_score"], reverse=True)

        ranked = self._filter_low_relevance_rows(
            rows=ranked,
            job_title=job_title,
            required_skills=required_skills,
            min_experience=min_experience,
        )
        ranked.sort(key=lambda x: x["predicted_fit_score"], reverse=True)

        self._log_prediction_distribution(
            job_title=job_title,
            required_skills=required_skills,
            rows=ranked,
        )
        self._log_prediction_and_calibration_drift(
            job_title=job_title,
            rows=ranked,
        )
        return ranked[:limit]

    def _rank_with_model(
        self,
        *,
        matcher: Any,
        job_title: str,
        required_skills: Sequence[str],
        min_experience: int,
        employees: Sequence[Any],
        model_source: str,
    ) -> list[dict[str, Any]]:
        job_payload = {
            "title": job_title,
            "required_skills": list(required_skills),
            "min_experience": min_experience,
        }

        job_key = self._job_cache_key(job_title, required_skills, min_experience)

        out: list[dict[str, Any]] = []
        pending: list[tuple[Any, str]] = []

        for employee in employees:
            cache_key = f"{model_source}|{job_key}|{self._employee_cache_key(employee)}"
            cached = self._cache_get(cache_key)
            if cached:
                out.append(cached)
            else:
                pending.append((employee, cache_key))

        preds: list[dict[str, Any]] | None = None
        if pending and hasattr(matcher, "predict_scores"):
            batch_start = perf_counter()
            try:
                preds = matcher.predict_scores(
                    [emp for emp, _ in pending],
                    job_payload,
                    batch_size=256,
                )
                latency_ms = (perf_counter() - batch_start) * 1000.0
                self._log_inference_metrics(
                    source="model",
                    job_title=job_title,
                    total=len(pending),
                    success=len(preds),
                    failed=max(0, len(pending) - len(preds)),
                    latency_ms=latency_ms,
                    batch=True,
                )
            except Exception:
                latency_ms = (perf_counter() - batch_start) * 1000.0
                self._log_inference_metrics(
                    source="model",
                    job_title=job_title,
                    total=len(pending),
                    success=0,
                    failed=len(pending),
                    latency_ms=latency_ms,
                    batch=True,
                )
                self._log_structured_failure(
                    reason=REASON_MODEL_FAIL,
                    stage="predict_scores_batch",
                    source="model",
                    job_title=job_title,
                )
                logger.exception(
                    "ai_prediction_failure scope=batch source=model job_title=%s",
                    job_title,
                )
                preds = None

        if pending and preds is None:
            for employee, cache_key in pending:
                try:
                    single_start = perf_counter()
                    pred = matcher.predict_score(employee, job_payload)
                    latency_ms = (perf_counter() - single_start) * 1000.0
                    self._log_inference_metrics(
                        source="model",
                        job_title=job_title,
                        total=1,
                        success=1,
                        failed=0,
                        latency_ms=latency_ms,
                        batch=False,
                    )

                    features = pred.get("features", {}) or {}
                    breakdown = self._model_feature_breakdown(matcher, features)
                    score = float(pred.get("score_percent", 0.0))
                    original_model_score = score

                    gap_report = build_training_recommendations(
                        job_title=job_title,
                        required_skills=required_skills,
                        owned_skills=self._extract_employee_skills(employee),
                        top_k=3,
                    )
                    experience_years = round(self._employee_experience_years(employee), 2)
                    overlap = self._bounded_unit(
                        features.get(
                            "skill_overlap",
                            1.0 - float(gap_report.get("skill_gap_ratio", 1.0)),
                        ),
                        default=0.0,
                    )
                    score, guardrail_reason = self._apply_required_skill_guardrail(
                        score_percent=score,
                        skill_overlap=overlap,
                        required_skills=required_skills,
                    )
                    score, experience_guardrail_reason = self._apply_experience_guardrail(
                        score_percent=score,
                        predicted_experience_years=experience_years,
                        min_experience=min_experience,
                    )
                    score = self._apply_hands_on_projects_bonus(
                        score_percent=score,
                        employee=employee,
                        required_skills=required_skills,
                        job_title=job_title,
                        breakdown=breakdown,
                    )
                    reasons = self._top_reasons(features, breakdown)
                    if guardrail_reason:
                        reasons = [guardrail_reason, *reasons]
                    if experience_guardrail_reason:
                        reasons = [experience_guardrail_reason, *reasons]
                    if (guardrail_reason or experience_guardrail_reason) and score < original_model_score:
                        breakdown["required_skill_guardrail"] = round(score - original_model_score, 6)

                    row = {
                        "employee_id": int(employee.id),
                        "full_name": self._full_name(employee),
                        "score": round(score, 2),
                        "predicted_fit_score": round(score, 2),
                        "predicted_title": (str(getattr(employee, "position", "")).strip() or None),
                        "predicted_experience_years": experience_years,
                        "score_raw": float(score),
                        "predicted_fit_score_raw": float(score),
                        "scoring_source": model_source,
                        "feature_breakdown": breakdown,
                        "top_reasons": reasons,
                        "matched_skills": gap_report["matched_skills"],
                        "skill_gaps": gap_report["missing_skills"],
                        "skill_gap_ratio": float(gap_report["skill_gap_ratio"]),
                        "learning_recommendations": gap_report["learning_recommendations"],
                    }
                    out.append(row)
                    self._cache_set(cache_key, row)

                except Exception:
                    emp_id = getattr(employee, "id", None)
                    self._log_structured_failure(
                        reason=REASON_MODEL_FAIL,
                        stage="predict_score_single",
                        source="model",
                        job_title=job_title,
                        employee_id=None if emp_id is None else int(emp_id),
                    )
                    logger.exception(
                        "ai_prediction_failure scope=employee source=model employee_id=%s job_title=%s",
                        emp_id,
                        job_title,
                    )
                    fallback = self._build_heuristic_row(
                        employee=employee,
                        job_title=job_title,
                        required_skills=required_skills,
                        min_experience=min_experience,
                        scoring_source="heuristic_fallback",
                    )
                    if fallback is not None:
                        self._log_structured_fallback(
                            stage="predict_score_single",
                            fallback_source="heuristic",
                            job_title=job_title,
                            employee_id=int(fallback["employee_id"]),
                            details="model_single_failed",
                        )
                        out.append(fallback)

        if preds is not None:
            for (employee, cache_key), pred in zip(pending, preds):
                try:
                    features = pred.get("features", {}) or {}
                    breakdown = self._model_feature_breakdown(matcher, features)
                    score = float(pred.get("score_percent", 0.0))
                    original_model_score = score

                    gap_report = build_training_recommendations(
                        job_title=job_title,
                        required_skills=required_skills,
                        owned_skills=self._extract_employee_skills(employee),
                        top_k=3,
                    )
                    experience_years = round(self._employee_experience_years(employee), 2)
                    overlap = self._bounded_unit(
                        features.get(
                            "skill_overlap",
                            1.0 - float(gap_report.get("skill_gap_ratio", 1.0)),
                        ),
                        default=0.0,
                    )
                    score, guardrail_reason = self._apply_required_skill_guardrail(
                        score_percent=score,
                        skill_overlap=overlap,
                        required_skills=required_skills,
                    )
                    score, experience_guardrail_reason = self._apply_experience_guardrail(
                        score_percent=score,
                        predicted_experience_years=experience_years,
                        min_experience=min_experience,
                    )
                    score = self._apply_hands_on_projects_bonus(
                        score_percent=score,
                        employee=employee,
                        required_skills=required_skills,
                        job_title=job_title,
                        breakdown=breakdown,
                    )
                    reasons = self._top_reasons(features, breakdown)
                    if guardrail_reason:
                        reasons = [guardrail_reason, *reasons]
                    if experience_guardrail_reason:
                        reasons = [experience_guardrail_reason, *reasons]
                    if (guardrail_reason or experience_guardrail_reason) and score < original_model_score:
                        breakdown["required_skill_guardrail"] = round(score - original_model_score, 6)

                    row = {
                        "employee_id": int(employee.id),
                        "full_name": self._full_name(employee),
                        "score": round(score, 2),
                        "predicted_fit_score": round(score, 2),
                        "predicted_title": (str(getattr(employee, "position", "")).strip() or None),
                        "predicted_experience_years": experience_years,
                        "score_raw": float(score),
                        "predicted_fit_score_raw": float(score),
                        "scoring_source": model_source,
                        "feature_breakdown": breakdown,
                        "top_reasons": reasons,
                        "matched_skills": gap_report["matched_skills"],
                        "skill_gaps": gap_report["missing_skills"],
                        "skill_gap_ratio": float(gap_report["skill_gap_ratio"]),
                        "learning_recommendations": gap_report["learning_recommendations"],
                    }
                    out.append(row)
                    self._cache_set(cache_key, row)

                except Exception:
                    emp_id = getattr(employee, "id", None)
                    self._log_structured_failure(
                        reason=REASON_MODEL_FAIL,
                        stage="predict_scores_row",
                        source="model",
                        job_title=job_title,
                        employee_id=None if emp_id is None else int(emp_id),
                    )
                    logger.exception(
                        "ai_prediction_failure scope=employee source=model employee_id=%s job_title=%s",
                        emp_id,
                        job_title,
                    )
                    fallback = self._build_heuristic_row(
                        employee=employee,
                        job_title=job_title,
                        required_skills=required_skills,
                        min_experience=min_experience,
                        scoring_source="heuristic_fallback",
                    )
                    if fallback is not None:
                        self._log_structured_fallback(
                            stage="predict_scores_row",
                            fallback_source="heuristic",
                            job_title=job_title,
                            employee_id=int(fallback["employee_id"]),
                            details="model_row_parse_or_feature_failure",
                        )
                        out.append(fallback)

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
            row = self._build_heuristic_row(
                employee=employee,
                job_title=job_title,
                required_skills=required_skills,
                min_experience=min_experience,
                scoring_source="heuristic",
            )
            if row is not None:
                out.append(row)
        return out

    def _scores_look_collapsed(self, rows: Sequence[dict[str, Any]]) -> bool:
        if len(rows) < 2:
            return False

        scores = [float(row.get("predicted_fit_score", 0.0)) for row in rows]
        rounded_scores = [round(score, 2) for score in scores]
        unique_count = len(set(rounded_scores))
        stddev = float(np.std(scores)) if scores else 0.0
        spread = (max(scores) - min(scores)) if scores else 0.0

        if unique_count <= self._COLLAPSED_SCORE_MAX_UNIQUE:
            return True
        if stddev <= self._COLLAPSED_SCORE_STDDEV_MAX:
            return True
        return spread <= 0.25

    def _recover_collapsed_scores(
        self,
        *,
        rows: Sequence[dict[str, Any]],
        employees: Sequence[Any],
        job_title: str,
        required_skills: Sequence[str],
        min_experience: int,
        model_source: str,
    ) -> list[dict[str, Any]]:
        by_id = {int(getattr(emp, "id")): emp for emp in employees if getattr(emp, "id", None) is not None}
        recovered: list[dict[str, Any]] = []

        for row in rows:
            emp_id = row.get("employee_id")
            if emp_id is None:
                continue
            employee = by_id.get(int(emp_id))
            if employee is None:
                continue

            h_row = self._build_heuristic_row(
                employee=employee,
                job_title=job_title,
                required_skills=required_skills,
                min_experience=min_experience,
                scoring_source="heuristic_recovery",
            )
            if h_row is None:
                continue

            model_score = float(row.get("predicted_fit_score", 0.0))
            heuristic_score = float(h_row.get("predicted_fit_score", 0.0))
            blended = round((0.2 * model_score) + (0.8 * heuristic_score), 2)

            merged = dict(row)
            merged["score"] = blended
            merged["predicted_fit_score"] = blended
            merged["score_raw"] = float(blended)
            merged["predicted_fit_score_raw"] = float(blended)
            merged["scoring_source"] = f"{model_source}+heuristic_recovery"
            merged["feature_breakdown"] = dict(h_row.get("feature_breakdown", {}))
            merged["top_reasons"] = list(h_row.get("top_reasons", []))
            recovered.append(merged)

        if not recovered:
            return []

        if len(recovered) < len(rows):
            recovered_ids = {int(item.get("employee_id")) for item in recovered if item.get("employee_id") is not None}
            for row in rows:
                emp_id = row.get("employee_id")
                if emp_id is None:
                    recovered.append(dict(row))
                    continue
                if int(emp_id) not in recovered_ids:
                    recovered.append(dict(row))

        logger.info(
            "ai_collapsed_scores_recovered job_title=%s total=%d source=%s",
            job_title,
            len(recovered),
            model_source,
        )
        return recovered

    def _build_heuristic_row(
        self,
        *,
        employee: Any,
        job_title: str,
        required_skills: Sequence[str],
        min_experience: int,
        scoring_source: str,
    ) -> dict[str, Any] | None:
        job_key = self._job_cache_key(job_title, required_skills, min_experience)
        cache_key = f"{scoring_source}|{job_key}|{self._employee_cache_key(employee)}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached

        try:
            h = calculate_weighted_score(
                employee=employee,
                job_title=job_title,
                required_skills=required_skills,
                min_experience=min_experience,
            )
        except Exception:
            logger.exception(
                "ai_prediction_failure scope=employee source=heuristic employee_id=%s job_title=%s",
                getattr(employee, "id", None),
                job_title,
            )
            return None

        score = float(h.get("total", 0.0))
        breakdown = {
            "skill_overlap": float(h.get("skill_overlap", 0.0)),
            "title_alignment": float(h.get("title_alignment", 0.0)),
            "experience_score": float(h.get("experience_score", 0.0)),
            "semantic_similarity": float(h.get("semantic_similarity", 0.0)),
            "performance_score": float(h.get("performance_score", 0.0)),
        }
        base_heuristic_score = score
        experience_years = round(self._employee_experience_years(employee), 2)
        score, guardrail_reason = self._apply_required_skill_guardrail(
            score_percent=score,
            skill_overlap=breakdown.get("skill_overlap", 0.0),
            required_skills=required_skills,
        )
        score, experience_guardrail_reason = self._apply_experience_guardrail(
            score_percent=score,
            predicted_experience_years=experience_years,
            min_experience=min_experience,
        )
        score = self._apply_hands_on_projects_bonus(
            score_percent=score,
            employee=employee,
            required_skills=required_skills,
            job_title=job_title,
            breakdown=breakdown,
        )
        reasons = self._top_reasons(breakdown, breakdown)
        if guardrail_reason:
            reasons = [guardrail_reason, *reasons]
        if experience_guardrail_reason:
            reasons = [experience_guardrail_reason, *reasons]
        if (guardrail_reason or experience_guardrail_reason) and score < base_heuristic_score:
            breakdown["required_skill_guardrail"] = round(score - base_heuristic_score, 6)

        gap_report = build_training_recommendations(
            job_title=job_title,
            required_skills=required_skills,
            owned_skills=self._extract_employee_skills(employee),
            top_k=3,
        )

        row = {
            "employee_id": int(employee.id),
            "full_name": self._full_name(employee),
            "score": round(score, 2),
            "predicted_fit_score": round(score, 2),
            "predicted_title": (str(getattr(employee, "position", "")).strip() or None),
            "predicted_experience_years": experience_years,
            "score_raw": float(score),
            "predicted_fit_score_raw": float(score),
            "scoring_source": scoring_source,
            "feature_breakdown": {k: round(v, 4) for k, v in breakdown.items()},
            "top_reasons": reasons,
            "matched_skills": gap_report["matched_skills"],
            "skill_gaps": gap_report["missing_skills"],
            "skill_gap_ratio": float(gap_report["skill_gap_ratio"]),
            "learning_recommendations": gap_report["learning_recommendations"],
        }
        self._cache_set(cache_key, row)
        return row

    def _log_prediction_distribution(
        self,
        *,
        job_title: str,
        required_skills: Sequence[str],
        rows: Sequence[dict[str, Any]],
    ) -> None:
        bins = {"0-20": 0, "20-40": 0, "40-60": 0, "60-80": 0, "80-100": 0}
        source_counts: dict[str, int] = {}

        if not rows:
            logger.info(
                "ai_prediction_distribution job_title=%s required_skills=%d total=0 bins=%s sources=%s",
                job_title,
                len(required_skills),
                bins,
                source_counts,
            )
            return

        scores: list[float] = []
        for row in rows:
            score = float(row.get("predicted_fit_score", 0.0))
            scores.append(score)

            if score < 20:
                bins["0-20"] += 1
            elif score < 40:
                bins["20-40"] += 1
            elif score < 60:
                bins["40-60"] += 1
            elif score < 80:
                bins["60-80"] += 1
            else:
                bins["80-100"] += 1

            source = str(row.get("scoring_source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1

        logger.info(
            "ai_prediction_distribution job_title=%s required_skills=%d total=%d mean=%.2f min=%.2f max=%.2f bins=%s sources=%s",
            job_title,
            len(required_skills),
            len(scores),
            float(np.mean(scores)),
            float(min(scores)),
            float(max(scores)),
            bins,
            source_counts,
        )

    def _log_prediction_and_calibration_drift(
        self,
        *,
        job_title: str,
        rows: Sequence[dict[str, Any]],
    ) -> None:
        if not self._DRIFT_MONITORING_ENABLED:
            return

        current_df = self._build_monitoring_frame(rows)
        if current_df.empty:
            logger.info("ai_drift_monitoring_skipped reason=no_scores job_title=%s", job_title)
            return

        self._append_current_monitoring_rows(job_title=job_title, current_df=current_df)
        calibration_warning = self._log_calibration_warning(job_title=job_title, current_df=current_df)

        if len(current_df) < self._DRIFT_MIN_ROWS:
            logger.info(
                "ai_drift_monitoring_skipped reason=insufficient_rows job_title=%s current=%d min_required=%d calibration_warning=%s",
                job_title,
                len(current_df),
                self._DRIFT_MIN_ROWS,
                calibration_warning,
            )
            return

        reference_df = self._load_reference_monitoring_frame()
        if reference_df is None or reference_df.empty:
            logger.warning(
                "ai_drift_monitoring_skipped reason=missing_reference baseline_path=%s",
                self._DRIFT_BASELINE_PATH,
            )
            return

        current_scores = current_df["predicted_score"].to_numpy(dtype=np.float64)
        reference_scores = reference_df["predicted_score"].to_numpy(dtype=np.float64)
        score_psi = self._population_stability_index(reference_scores, current_scores, bins=10)

        ref_ece = self._expected_calibration_error(reference_df)
        cur_ece = self._expected_calibration_error(current_df)
        calibration_drift = None
        if not np.isnan(ref_ece) and not np.isnan(cur_ece):
            calibration_drift = float(cur_ece - ref_ece)

        report_path = self._write_whylogs_profile(
            current_df=current_df,
            job_title=job_title,
        )
        self._append_drift_event(
            job_title=job_title,
            reference_rows=len(reference_df),
            current_rows=len(current_df),
            score_psi=score_psi,
            ref_ece=ref_ece,
            cur_ece=cur_ece,
            calibration_drift=calibration_drift,
            calibration_warning=calibration_warning,
            whylogs_profile=report_path,
        )

        logger.info(
            "ai_drift_metrics job_title=%s reference_rows=%d current_rows=%d score_psi=%.6f ref_ece=%s cur_ece=%s calibration_drift=%s calibration_warning=%s whylogs_profile=%s",
            job_title,
            len(reference_df),
            len(current_df),
            score_psi,
            "nan" if np.isnan(ref_ece) else f"{ref_ece:.6f}",
            "nan" if np.isnan(cur_ece) else f"{cur_ece:.6f}",
            "nan" if calibration_drift is None else f"{calibration_drift:.6f}",
            calibration_warning,
            report_path or "none",
        )

    def _build_monitoring_frame(self, rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
        out_rows: list[dict[str, Any]] = []
        for row in rows:
            raw_score = row.get(
                "predicted_fit_score_raw",
                row.get("score_raw", row.get("predicted_fit_score", row.get("score", 0.0))),
            )
            score = self._normalize_score_01(raw_score)
            label = self._extract_optional_label(row)
            out_rows.append(
                {
                    "predicted_score": score,
                    "scoring_source": str(row.get("scoring_source", "unknown")),
                    "actual_label": label,
                }
            )

        df = pd.DataFrame(out_rows)
        if df.empty:
            return df

        df["predicted_score"] = pd.to_numeric(df["predicted_score"], errors="coerce").fillna(0.0)
        df["predicted_score"] = df["predicted_score"].clip(0.0, 1.0)
        return df

    def _load_reference_monitoring_frame(self) -> pd.DataFrame | None:
        path = self._DRIFT_BASELINE_PATH
        if not path.exists():
            return None

        try:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                ref = pd.read_csv(path)
            elif suffix == ".parquet":
                ref = pd.read_parquet(path)
            elif suffix in {".jsonl", ".ndjson"}:
                ref = pd.read_json(path, lines=True)
            else:
                logger.warning("ai_drift_reference_unsupported suffix=%s path=%s", suffix, path)
                return None
        except Exception:
            logger.exception("ai_drift_reference_load_failure path=%s", path)
            return None

        if "predicted_score" not in ref.columns and "predicted_fit_score" in ref.columns:
            ref["predicted_score"] = ref["predicted_fit_score"]
        if "predicted_score" not in ref.columns and "score" in ref.columns:
            ref["predicted_score"] = ref["score"]
        if "predicted_score" not in ref.columns:
            logger.warning("ai_drift_reference_missing_column path=%s column=predicted_score", path)
            return None

        ref_df = pd.DataFrame(
            {
                "predicted_score": pd.to_numeric(ref["predicted_score"], errors="coerce"),
                "scoring_source": ref.get("scoring_source", "reference"),
                "actual_label": ref.get("actual_label", ref.get("label")),
            }
        )

        ref_df = ref_df.dropna(subset=["predicted_score"]).copy()
        if ref_df.empty:
            return None

        ref_df["predicted_score"] = ref_df["predicted_score"].map(self._normalize_score_01).clip(0.0, 1.0)
        ref_df["actual_label"] = ref_df["actual_label"].map(self._to_binary_label)
        return ref_df

    def _write_whylogs_profile(
        self,
        *,
        current_df: pd.DataFrame,
        job_title: str,
    ) -> str | None:
        try:
            if not hasattr(np, "unicode_"):
                np.unicode_ = np.str_
            import whylogs as why
        except Exception:
            logger.warning("ai_whylogs_unavailable package=whylogs")
            return None

        try:
            self._WHYLOGS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            safe_title = (job_title or "unknown").strip().lower().replace(" ", "_")[:64] or "unknown"
            out_path = self._WHYLOGS_REPORTS_DIR / f"profile_{safe_title}_{stamp}.bin"

            profile_df = current_df[["predicted_score"]].copy()
            profile = why.log(pandas=profile_df).profile()
            profile.view().writer("local").write(dest=str(out_path))
            return str(out_path)
        except Exception:
            logger.exception("ai_whylogs_write_failure")
            return None

    def _append_current_monitoring_rows(self, *, job_title: str, current_df: pd.DataFrame) -> None:
        path = self._MONITORING_EVENTS_PATH
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not path.exists() or path.stat().st_size == 0
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            with path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "scored_at_utc",
                        "job_title",
                        "predicted_score",
                        "actual_label",
                        "scoring_source",
                    ],
                )
                if write_header:
                    writer.writeheader()
                for row in current_df.itertuples(index=False):
                    writer.writerow(
                        {
                            "scored_at_utc": stamp,
                            "job_title": (job_title or "").strip(),
                            "predicted_score": float(getattr(row, "predicted_score", 0.0)),
                            "actual_label": getattr(row, "actual_label", np.nan),
                            "scoring_source": str(getattr(row, "scoring_source", "unknown")),
                        }
                    )
        except Exception:
            logger.exception("ai_monitoring_rows_persist_failure path=%s", path)

    def _log_calibration_warning(self, *, job_title: str, current_df: pd.DataFrame) -> bool:
        scores = current_df["predicted_score"].to_numpy(dtype=np.float64)
        if scores.size == 0:
            return False

        mean_weighted = float(np.mean(scores))
        stddev_weighted = float(np.std(scores))
        warning = (mean_weighted > self._CALIBRATION_WARNING_MEAN_GT) or (
            stddev_weighted < self._CALIBRATION_WARNING_STD_LT
        )

        if warning:
            logger.warning(
                "ai_calibration_warning job_title=%s mean_weighted=%.6f stddev_weighted=%.6f mean_threshold=%.6f std_threshold=%.6f",
                job_title,
                mean_weighted,
                stddev_weighted,
                self._CALIBRATION_WARNING_MEAN_GT,
                self._CALIBRATION_WARNING_STD_LT,
            )
        return warning

    def _append_drift_event(
        self,
        *,
        job_title: str,
        reference_rows: int,
        current_rows: int,
        score_psi: float,
        ref_ece: float,
        cur_ece: float,
        calibration_drift: float | None,
        calibration_warning: bool,
        whylogs_profile: str | None,
    ) -> None:
        path = self._DRIFT_EVENTS_PATH
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "job_title": (job_title or "").strip(),
            "reference_rows": int(reference_rows),
            "current_rows": int(current_rows),
            "score_psi": None if np.isnan(score_psi) else float(score_psi),
            "ref_ece": None if np.isnan(ref_ece) else float(ref_ece),
            "cur_ece": None if np.isnan(cur_ece) else float(cur_ece),
            "calibration_drift": None if calibration_drift is None else float(calibration_drift),
            "calibration_warning": bool(calibration_warning),
            "whylogs_profile": whylogs_profile,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception:
            logger.exception("ai_drift_event_persist_failure path=%s", path)

    def _normalize_score_01(self, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        if score > 1.0:
            score = score / 100.0
        return float(min(1.0, max(0.0, score)))

    def _extract_optional_label(self, row: dict[str, Any]) -> float | np.floating[Any] | None:
        for key in ("actual_label", "label", "outcome", "is_match"):
            if key in row:
                return self._to_binary_label(row.get(key))
        return None

    def _to_binary_label(self, value: Any) -> float | np.floating[Any]:
        if value is None:
            return np.nan
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return 1.0 if float(value) >= 1.0 else 0.0
        if isinstance(value, str):
            raw = value.strip().lower()
            if raw in {"1", "true", "yes", "y", "positive", "match"}:
                return 1.0
            if raw in {"0", "false", "no", "n", "negative", "no_match"}:
                return 0.0
        return np.nan

    def _population_stability_index(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        if reference.size == 0 or current.size == 0:
            return float("nan")

        edges = np.linspace(0.0, 1.0, bins + 1)
        ref_hist, _ = np.histogram(reference, bins=edges)
        cur_hist, _ = np.histogram(current, bins=edges)

        ref_ratio = ref_hist / max(ref_hist.sum(), 1)
        cur_ratio = cur_hist / max(cur_hist.sum(), 1)

        eps = 1e-9
        ref_ratio = np.clip(ref_ratio, eps, None)
        cur_ratio = np.clip(cur_ratio, eps, None)
        return float(np.sum((cur_ratio - ref_ratio) * np.log(cur_ratio / ref_ratio)))

    def _expected_calibration_error(self, df: pd.DataFrame) -> float:
        if "actual_label" not in df.columns:
            return float("nan")

        valid = df.dropna(subset=["actual_label"]).copy()
        if valid.empty:
            return float("nan")

        probs = valid["predicted_score"].to_numpy(dtype=np.float64)
        labels = valid["actual_label"].to_numpy(dtype=np.float64)

        edges = np.linspace(0.0, 1.0, self._CALIBRATION_BINS + 1)
        ece = 0.0
        total = len(probs)

        for start, end in zip(edges[:-1], edges[1:]):
            if end >= 1.0:
                mask = (probs >= start) & (probs <= end)
            else:
                mask = (probs >= start) & (probs < end)

            if not np.any(mask):
                continue

            bin_probs = probs[mask]
            bin_labels = labels[mask]
            ece += (bin_probs.size / total) * abs(float(np.mean(bin_probs)) - float(np.mean(bin_labels)))

        return float(ece)

    def _model_feature_breakdown(self, matcher: Any, features: dict[str, float]) -> dict[str, float]:
        # Keep explanation components human-interpretable and directionally stable.
        skill_overlap = self._bounded_unit(features.get("skill_overlap", 0.0), default=0.0)
        missing_ratio = self._bounded_unit(features.get("missing_skill_ratio", 0.0), default=0.0)
        semantic_similarity = self._bounded_unit(features.get("semantic_similarity", 0.0), default=0.0)
        performance_score = self._bounded_unit(features.get("performance_score", 0.0), default=0.0)
        engagement_score = self._bounded_unit(features.get("engagement_score", 0.0), default=0.0)
        satisfaction_score = self._bounded_unit(features.get("satisfaction_score", 0.0), default=0.0)
        currently_active = self._bounded_unit(features.get("currently_active", 0.0), default=0.0)

        try:
            exp_gap_raw = max(0.0, float(features.get("experience_gap", 0.0)))
        except (TypeError, ValueError):
            exp_gap_raw = 0.0
        try:
            exp_surplus_raw = max(0.0, float(features.get("experience_surplus", 0.0)))
        except (TypeError, ValueError):
            exp_surplus_raw = 0.0

        exp_gap = min(exp_gap_raw / 5.0, 1.0)
        exp_surplus = min(exp_surplus_raw / 5.0, 1.0)

        return {
            "skill_overlap": round(skill_overlap, 6),
            "missing_skill_ratio": round(-missing_ratio, 6),
            "experience_gap": round(-exp_gap, 6),
            "experience_surplus": round(exp_surplus, 6),
            "semantic_similarity": round(semantic_similarity, 6),
            "performance_score": round(performance_score, 6),
            "engagement_score": round(engagement_score, 6),
            "satisfaction_score": round(satisfaction_score, 6),
            "currently_active": round(0.2 * currently_active, 6),
        }

    def _extract_employee_projects(self, employee: Any) -> list[str]:
        raw = getattr(employee, "candidate_projects", None)
        if not raw:
            return []
        if isinstance(raw, list):
            values = [str(item or "").strip() for item in raw]
            return [item for item in values if item]
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return []
            try:
                decoded = json.loads(text)
            except Exception:
                return []
            if not isinstance(decoded, list):
                return []
            values = [str(item or "").strip() for item in decoded]
            return [item for item in values if item]
        return []

    def _project_relevance_score(
        self,
        *,
        projects: Sequence[str],
        required_skills: Sequence[str],
        job_title: str,
    ) -> float:
        if not projects:
            return 0.0
        skill_tokens = {
            normalize_skill_name(s)
            for s in (required_skills or [])
            if normalize_skill_name(s)
        }
        title_tokens = {
            tok for tok in normalize_skill_name(job_title).split() if tok
        }
        target_tokens = set(skill_tokens) | set(title_tokens)
        if not target_tokens:
            return 0.0

        matched_projects = 0
        keyword_hits: set[str] = set()
        for project in projects:
            norm = normalize_skill_name(project)
            if not norm:
                continue
            tokens = {tok for tok in re.split(r"\s+", norm) if tok}
            overlap = tokens & target_tokens
            if overlap:
                matched_projects += 1
                keyword_hits.update(overlap)

        if matched_projects == 0:
            return 0.0

        breadth = min(len(keyword_hits) / max(1, len(skill_tokens) or len(target_tokens)), 1.0)
        volume = min(matched_projects / 4.0, 1.0)
        return float(max(0.0, min(1.0, (0.65 * breadth) + (0.35 * volume))))

    def _apply_hands_on_projects_bonus(
        self,
        *,
        score_percent: float,
        employee: Any,
        required_skills: Sequence[str],
        job_title: str,
        breakdown: dict[str, float],
    ) -> float:
        projects = self._extract_employee_projects(employee)
        relevance = self._project_relevance_score(
            projects=projects,
            required_skills=required_skills,
            job_title=job_title,
        )
        bonus = relevance * max(0.0, self._PROJECT_BONUS_MAX)
        breakdown["hands_on_projects"] = round(float(bonus), 6)
        return float(min(100.0, max(0.0, float(score_percent) + bonus)))

    def _top_reasons(self, features: dict[str, float], breakdown: dict[str, float], top_n: int = 3) -> list[str]:
        items = list(breakdown.items())
        positives = [(k, v) for k, v in items if v > 0]
        ordered = sorted(positives, key=lambda x: x[1], reverse=True) or sorted(
            items, key=lambda x: abs(x[1]), reverse=True
        )
        return [self._reason_text(name, float(features.get(name, 0.0)), float(val)) for name, val in ordered[:top_n]]

    def _reason_text(self, feature: str, value: float, contribution: float) -> str:
        if feature == "skill_overlap":
            return f"Strong skill overlap ({value:.2f})."
        if feature == "title_alignment":
            return f"Job title alignment ({value:.2f})."
        if feature == "experience_surplus":
            return f"Experience exceeds requirement ({value:.2f} years)."
        if feature == "experience_gap":
            return f"Low experience gap ({value:.2f} years)."
        if feature == "semantic_similarity":
            return f"High semantic similarity ({value:.2f})."
        if feature == "performance_score":
            return f"Good performance history ({value:.2f})."
        if feature == "hands_on_projects":
            return f"Relevant hands-on projects added value ({value:.2f})."
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

    def _employee_experience_years(self, employee: Any) -> float:
        hire_date = getattr(employee, "hire_date", None)
        if hire_date is None:
            return 0.0
        days = max(0, (datetime.now(timezone.utc).date() - hire_date).days)
        return float(days / 365.25)

    def _extract_employee_skills(self, employee: Any) -> list[str]:
        names: list[str] = []
        for item in getattr(employee, "skills", []) or []:
            if hasattr(item, "name") and getattr(item, "name", None):
                names.append(str(item.name))
                continue

            linked_skill = getattr(item, "skill", None)
            linked_name = getattr(linked_skill, "name", None)
            if linked_name:
                names.append(str(linked_name))
        return names
