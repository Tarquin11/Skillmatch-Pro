import hashlib
from sqlalchemy.inspection import inspect
from datetime import datetime, timezone
from typing import Any
from fastapi import HTTPException, Response, status

def _norm_value(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat(timespec="microseconds")
    return str(value)

def _entity_fingerprint(entity: Any) -> str:
    mapper = inspect(entity).mapper
    parts: list[str] = []
    for attr in mapper.column_attrs:
        key = attr.key
        if key == "updated_at":
            continue
        parts.append(f"{key}={_norm_value(getattr(entity, key, None))}")
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]

def _normalize_etag_token(token: str) -> str:
    token = token.strip()
    if token.startswith("W/"):
        token = token[2:].strip()
    return token

def build_etag(entity: Any) -> str:
    entity_id = getattr(entity, "id", None)
    updated_at = getattr(entity, "updated_at", None)

    if isinstance(updated_at,datetime):
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        stamp = updated_at.astimezone(timezone.utc).isoformat(timespec="microseconds")
    else:
        stamp = "0"

    fp = _entity_fingerprint(entity)
    return f"\"{entity_id}:{stamp}:{fp}\""

def set_etag(response: Response, entity: Any) -> str:
    etag = build_etag(entity)
    response.headers["ETag"] = etag
    return etag

def enforce_if_match(entity: Any, if_match: str | None) -> None:
    if not if_match:
        raise HTTPException(status_code=status.HTTP_428_PRECONDITION_REQUIRED, detail={"code":"precondition_required", "message": "Missing if-match header"},)
    tokens = {_normalize_etag_token(part) for part in if_match.split(",")if part.strip()}
    if "*" in tokens:
        return
    
    current = build_etag(entity)
    if current not in tokens:
        raise HTTPException(
            status_code=status.HTTP_412_PRECONDITION_FAILED,
            detail={
                "code": "etag_mismatch",
                "message": "Resource has been modified",
            },
        )