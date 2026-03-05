from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import joblib
import numpy as np
import os
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,ndcg_score,precision_recall_fscore_support,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from app.ai.feature_engineering import FEATURE_COLUMNS, FeatureEngineer
from app.ai.preprocessing import preprocess_employee, preprocess_job, preprocess_training_pairs


@dataclass
class TrainResult:
    train_size: int
    valid_size: int
    roc_auc: float | None
    average_precision: float | None
    precision: float
    recall: float
    f1: float
    precision_at_k: float | None = None
    recall_at_k: float | None = None
    map_at_k: float | None = None
    ndcg_at_k: float | None = None


class CandidateMatcher:

    MODEL_NAME = "candidate_matcher"
    ARTIFACT_SCHEMA_VERSION = "1.0.0"

    def __init__(self, use_semantic: bool = True):
        self.feature_engineer = FeatureEngineer(use_semantic=use_semantic)
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1500, class_weight="balanced")),
            ]
        )
        self.is_fitted = False

    def _build_artifact_metadata(self, artifact_path: Path | None = None) -> dict[str, Any]:
        clf= None
        if hasattr(self.model, "named_steps"):
            clf = self.model.named_steps.get("clf")
            return {
                "model_name": self.MODEL_NAME,
                "model_version": os.getenv("AI_MODEL_VERSION", "dev"),
                "schema_version":self.ARTIFACT_SCHEMA_VERSION,
                "trained_at_utc":datetime.now(timezone.utc).isoformat(),
                "is_fitted": bool(self.is_fitted),
                "use_semantic":bool(self.feature_engineer.use_semantic),
                "feature_columns":list(FEATURE_COLUMNS),
                "classifier":type(clf).__name__ if clf is not None else type(self.model.__name__),
                "artifact_path": str(artifact_path) if artifact_path else None, 
            }
    @classmethod
    def read_artifact_metadata(cls, path: str | Path) -> dict[str, Any]:
        try: 
            payload = joblib.load(Path(path))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        metadata = payload.get("metadata")
        if isinstance(metadata, dict) and metadata:
            return dict(metadata)
        
        model = payload.get("model")
        return{
                "model_name":  cls.MODEL_NAME,
                "schema_version": "legacy",
                "is_fitted": "legacy",
                "use_semantic":bool(payload.get("is_fitted", False)),
                "feature_columns" : list(payload.get("feature_columns", FEATURE_COLUMNS)),
                "classifier": type(model).__name__ if model is not None else None,
                "artifact_path": str(Path(path)),
        }
    @staticmethod
    def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
        y_pred = (y_prob >= threshold).astype(np.int32)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        return float(precision), float(recall), float(f1)

    @staticmethod
    def _ranking_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        query_ids: Sequence[Any] | None,
        k: int = 10,
    ) -> dict[str, float | None]:
        if not query_ids or len(query_ids) != len(y_true):
            return {
                "precision_at_k": None,
                "recall_at_k": None,
                "map_at_k": None,
                "ndcg_at_k": None,
            }

        groups: dict[Any, list[int]] = {}
        for idx, q in enumerate(query_ids):
            groups.setdefault(q, []).append(idx)

        p_vals: list[float] = []
        r_vals: list[float] = []
        map_vals: list[float] = []
        ndcg_vals: list[float] = []

        for _, idxs in groups.items():
            true = y_true[idxs].astype(np.int32)
            score = y_prob[idxs].astype(np.float32)
            positives = int(true.sum())
            if positives == 0:
                continue

            order = np.argsort(-score)
            top = order[: min(k, len(order))]
            rel_top = true[top]

            p_at_k = float(rel_top.sum() / len(top))
            r_at_k = float(rel_top.sum() / positives)

            hits = 0
            precisions = []
            for rank, row_idx in enumerate(top, start=1):
                if true[row_idx] == 1:
                    hits += 1
                    precisions.append(hits / rank)
            ap_at_k = float(sum(precisions) / min(positives, len(top))) if precisions else 0.0

            if len(true) < 2 :
                ndcg_k = 1.0
            else:
                ndcg_k = float(ndcg_score([true],[score], k=min(k, len(true))))

            p_vals.append(p_at_k)
            r_vals.append(r_at_k)
            map_vals.append(ap_at_k)
            ndcg_vals.append(ndcg_k)

        if not p_vals:
            return {
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "map_at_k": 0.0,
                "ndcg_at_k": 0.0,
            }

        return {
            "precision_at_k": float(np.mean(p_vals)),
            "recall_at_k": float(np.mean(r_vals)),
            "map_at_k": float(np.mean(map_vals)),
            "ndcg_at_k": float(np.mean(ndcg_vals)),
        }

    def train(
        self,
        raw_pairs: list[dict[str, Any]],
        query_ids: Sequence[Any] | None = None,
        valid_size: float = 0.2,
        random_state: int = 42,
        k: int = 10,
    ) -> TrainResult:
        pairs = preprocess_training_pairs(raw_pairs)
        x, y = self.feature_engineer.vectorize_pairs(pairs)

        if y is None or len(y) == 0:
            raise ValueError("No labels provided for training.")
        if len(np.unique(y)) < 2:
            raise ValueError("Training requires at least two classes (0 and 1).")

        idx = np.arange(len(y))
        train_idx, valid_idx = train_test_split(
            idx,
            test_size=valid_size,
            random_state=random_state,
            stratify=y,
        )

        x_train, y_train = x[train_idx], y[train_idx]
        x_valid, y_valid = x[valid_idx], y[valid_idx]
        q_valid = [query_ids[i] for i in valid_idx] if query_ids else None

        self.model.fit(x_train, y_train)
        self.is_fitted = True

        y_prob = self.model.predict_proba(x_valid)[:, 1]
        roc_auc = roc_auc_score(y_valid, y_prob) if len(np.unique(y_valid)) > 1 else None
        avg_pr = average_precision_score(y_valid, y_prob) if len(np.unique(y_valid)) > 1 else None
        precision, recall, f1 = self._classification_metrics(y_valid, y_prob, threshold=0.5)
        rank = self._ranking_metrics(y_valid, y_prob, q_valid, k=k)

        return TrainResult(
            train_size=len(train_idx),
            valid_size=len(valid_idx),
            roc_auc=float(roc_auc) if roc_auc is not None else None,
            average_precision=float(avg_pr) if avg_pr is not None else None,
            precision=precision,
            recall=recall,
            f1=f1,
            precision_at_k=rank["precision_at_k"],
            recall_at_k=rank["recall_at_k"],
            map_at_k=rank["map_at_k"],
            ndcg_at_k=rank["ndcg_at_k"],
        )

    def _heuristic_score(self, features: dict[str, float]) -> float:
        score = (
            0.35 * features["skill_overlap"]
            + 0.20 * features["semantic_similarity"]
            + 0.15 * min(features["experience_surplus"] / 5.0, 1.0)
            + 0.10 * (1.0 - min(features["experience_gap"] / 5.0, 1.0))
            + 0.10 * features["performance_score"]
            + 0.05 * features["engagement_score"]
            + 0.05 * features["satisfaction_score"]
        )
        return float(max(0.0, min(1.0, score)))

    def predict_score(self, employee_raw: Any, job_raw: Any) -> dict[str, Any]:
        employee = preprocess_employee(employee_raw)
        job = preprocess_job(job_raw)

        features = self.feature_engineer.create_features(employee, job)
        row = np.asarray([[features[col] for col in FEATURE_COLUMNS]], dtype=np.float32)

        if self.is_fitted:
            score = float(self.model.predict_proba(row)[0, 1])
            source = "model"
        else:
            score = self._heuristic_score(features)
            source = "heuristic"

        return {
            "employee_id": employee.get("id"),
            "score": round(score, 6),
            "score_percent": round(score * 100, 2),
            "source": source,
            "features": features,
        }

    def rank_candidates(self, job_raw: Any, employee_list: list[Any], top_k: int = 20) -> list[dict[str, Any]]:
        scored = [self.predict_score(emp, job_raw) for emp in employee_list]
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "is_fitted": self.is_fitted,
            "model": self.model,
            "feature_columns": FEATURE_COLUMNS,
            "use_semantic": self.feature_engineer.use_semantic,
            "metadata" : self._build_artifact_metadata(path),
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "CandidateMatcher":
        payload = joblib.load(Path(path))
        matcher = cls(use_semantic=bool(payload.get("use_semantic", True)))
        matcher.model = payload["model"]
        matcher.is_fitted = bool(payload.get("is_fitted", False))
        return matcher
