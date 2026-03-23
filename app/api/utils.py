from __future__ import annotations
from typing import Any, Mapping
from fastapi import HTTPException
from sqlalchemy.orm import Query

def apply_list_query(
    query: Query,
    *,
    sort_by: str,
    sort_dir: str,
    sort_map: Mapping[str, Any],
    skip: int,
    limit: int,
) -> Query:
    sort_col = sort_map.get(sort_by)
    if sort_col is None:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_sort_field", "message": f"Invalid sort_by: {sort_by}"},
        )
    if sort_dir.lower() == "desc":
        query = query.order_by(sort_col.desc())
    else:
        query = query.order_by(sort_col.asc())

    return query.offset(skip).limit(limit)
