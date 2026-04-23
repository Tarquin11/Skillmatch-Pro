from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.auth import get_current_active_user, require_policy
from app.core.rbac import Policy
from app.db.database import get_db
from app.models.active_learning import UnknownEntity
from app.models.user import User
from app.schemas.active_learning import EntityReviewCreate, UnknownEntityOut
from app.services.active_learning import review_unknown_entity, serialize_unknown_entity


router = APIRouter(
    prefix="/learning",
    tags=["learning"],
    dependencies=[Depends(get_current_active_user)],
)


@router.get("/unknown-entities", response_model=list[UnknownEntityOut])
def list_unknown_entities(
    status_filter: str | None = Query(default=None, alias="status"),
    entity_type: str | None = Query(default=None),
    search: str | None = Query(default=None),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    include_reviews: bool = Query(default=False),
    db: Session = Depends(get_db),
    _current_user: User = Depends(require_policy(Policy.LEARNING_REVIEW)),
):
    query = db.query(UnknownEntity)
    if status_filter:
        query = query.filter(UnknownEntity.status == status_filter.strip().lower())
    if entity_type:
        query = query.filter(UnknownEntity.entity_type_guess == entity_type.strip().lower())
    if search:
        term = search.strip().lower()
        query = query.filter(
            func.lower(func.coalesce(UnknownEntity.raw_value, "")).like(f"%{term}%")
        )
    rows = (
        query.order_by(UnknownEntity.updated_at.desc(), UnknownEntity.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [UnknownEntityOut(**serialize_unknown_entity(row, include_reviews=include_reviews)) for row in rows]


@router.get("/unknown-entities/{entity_id}", response_model=UnknownEntityOut)
def get_unknown_entity(
    entity_id: int,
    db: Session = Depends(get_db),
    _current_user: User = Depends(require_policy(Policy.LEARNING_REVIEW)),
):
    entity = db.get(UnknownEntity, entity_id)
    if entity is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "unknown_entity_not_found", "message": "Unknown entity not found."},
        )
    return UnknownEntityOut(**serialize_unknown_entity(entity, include_reviews=True))


@router.post("/unknown-entities/{entity_id}/review", response_model=UnknownEntityOut)
def review_pending_unknown_entity(
    entity_id: int,
    payload: EntityReviewCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_policy(Policy.LEARNING_REVIEW)),
):
    entity = db.get(UnknownEntity, entity_id)
    if entity is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "unknown_entity_not_found", "message": "Unknown entity not found."},
        )

    updated = review_unknown_entity(
        db=db,
        entity=entity,
        reviewer_id=int(current_user.id),
        decision=payload.decision,
        entity_type=payload.entity_type,
        canonical_value=payload.canonical_value,
        notes=payload.notes,
    )
    return UnknownEntityOut(**serialize_unknown_entity(updated, include_reviews=True))
