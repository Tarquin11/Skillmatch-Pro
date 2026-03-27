from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from typing import Any

REASON_PARSING_FAIL = "parsing_fail"
REASON_MODEL_FAIL = "model_fail"
REASON_FALLBACK_USED = "fallback_used"
EVENT_AI_PREDICTION_FAILURE = "ai_prediction_failure"
EVENT_AI_FALLBACK_USED = "ai_fallback_used"
EVENT_CV_PARSE_FAILURE = "cv_parse_failure"
EVENT_CV_PARSE_FALLBACK = "cv_parse_fallback"

def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds")
    return str(value)

def log_structured_event(logger: logging.Logger,*,level: int,event: str,reason: str | None = None,**fields: Any,) -> None:
    payload: dict[str, Any] = {
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
    }
    if reason:
        payload["reason"] = reason

    for key, value in fields.items():
        if value is not None:
            payload[key] = value
    logger.log(
        level,
        json.dumps(payload, ensure_ascii=True, sort_keys=True, default=_json_default),
    )
