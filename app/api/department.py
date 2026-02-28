from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.auth import get_current_active_user, require_roles
from app.db.database import get_db
from app.models.departement import Departement
from app.models.user import User
from app.schemas.departement import DepartementCreate, DepartementOut, DepartementUpdate


router = APIRouter(
    prefix="/departements",
    tags=["departements"],
    dependencies=[Depends(get_current_active_user)],
)


def _normalize_name(value: str) -> str:
    return " ".join(value.split()).strip()


@router.get("/", response_model=List[DepartementOut])
def list_departements(
    search: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    query = db.query(Departement)
    if search:
        term = search.strip().lower()
        query = query.filter(func.lower(func.trim(Departement.name)).like(f"%{term}%"))
    return query.order_by(Departement.name.asc()).all()


@router.get("/{departement_id}", response_model=DepartementOut)
def get_departement(departement_id: int, db: Session = Depends(get_db)):
    departement = db.get(Departement, departement_id)
    if not departement:
        raise HTTPException(
            status_code=404,
            detail={"code": "departement_not_found", "message": "Departement not found"},
        )
    return departement


@router.post("/", response_model=DepartementOut, status_code=status.HTTP_201_CREATED)
def create_departement(
    payload: DepartementCreate,
    db: Session = Depends(get_db),
    _current_user: User = Depends(require_roles("admin")),
):
    name = _normalize_name(payload.name)
    if not name:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_departement_name", "message": "Departement name cannot be empty"},
        )

    existing = (
        db.query(Departement)
        .filter(func.lower(func.trim(Departement.name)) == name.lower())
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail={"code": "departement_already_exists", "message": "Departement already exists"},
        )

    departement = Departement(name=name, description=payload.description)
    db.add(departement)
    db.commit()
    db.refresh(departement)
    return departement


@router.put("/{departement_id}", response_model=DepartementOut)
def update_departement(
    departement_id: int,
    payload: DepartementUpdate,
    db: Session = Depends(get_db),
    _current_user: User = Depends(require_roles("admin")),
):
    departement = db.get(Departement, departement_id)
    if not departement:
        raise HTTPException(
            status_code=404,
            detail={"code": "departement_not_found", "message": "Departement not found"},
        )

    updates = payload.model_dump(exclude_unset=True)
    if "name" in updates and updates["name"] is not None:
        normalized_name = _normalize_name(updates["name"])
        if not normalized_name:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_departement_name", "message": "Departement name cannot be empty"},
            )

        conflict = (
            db.query(Departement)
            .filter(
                func.lower(func.trim(Departement.name)) == normalized_name.lower(),
                Departement.id != departement_id,
            )
            .first()
        )
        if conflict:
            raise HTTPException(
                status_code=409,
                detail={"code": "departement_already_exists", "message": "Departement already exists"},
            )

        updates["name"] = normalized_name

    for field, value in updates.items():
        setattr(departement, field, value)

    db.commit()
    db.refresh(departement)
    return departement


@router.delete("/{departement_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_departement(
    departement_id: int,
    db: Session = Depends(get_db),
    _current_user: User = Depends(require_roles("admin")),
):
    departement = db.get(Departement, departement_id)
    if not departement:
        raise HTTPException(
            status_code=404,
            detail={"code": "departement_not_found", "message": "Departement not found"},
        )

    db.delete(departement)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
