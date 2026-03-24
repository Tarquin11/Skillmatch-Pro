from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status, Header
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.api.auth import get_current_active_user, require_policy
from app.db.database import get_db
from app.models.departement import Departement
from app.models.user import User
from app.schemas.departement import DepartementCreate, DepartementOut, DepartementUpdate
from app.api.concurrency import enforce_if_match, set_etag
from app.core.rbac import Policy

router = APIRouter(tags=["departments"],dependencies=[Depends(get_current_active_user)],)

_DEPARTMENT_SORT_FIELDS = {
    "id": Departement.id,
    "name": Departement.name,
}

def _normalize_name(value: str) -> str:
    return " ".join(value.split()).strip()

@router.get("/", response_model=List[DepartementOut])
def list_departments(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(default=None),
    sort_by: str = Query("name"),
    sort_dir: str = Query("asc"),
    db: Session = Depends(get_db),
):
    query = db.query(Departement)
    if search:
        term = search.strip().lower()
        query = query.filter(func.lower(func.trim(Departement.name)).like(f"%{term}%"))
    sort_col = _DEPARTMENT_SORT_FIELDS.get(sort_by, Departement.name)
    if sort_dir.lower() == "desc":
        query = query.order_by(sort_col.desc())
    else:
        query = query.order_by(sort_col.asc())
    return query.offset(skip).limit(limit).all()

@router.get("/{department_id}", response_model=DepartementOut)
def get_department(department_id: int, response: Response, db: Session = Depends(get_db)):
    departement = db.get(Departement, department_id)
    if not departement:
        raise HTTPException(
            status_code=404,
            detail={"code": "department_not_found", "message": "Department not found"},
        )
    set_etag(response, departement)
    return departement

@router.post("/", response_model=DepartementOut, status_code=status.HTTP_201_CREATED)
def create_department(
    payload: DepartementCreate,
    db: Session = Depends(get_db),
    _current_user: User = Depends(require_policy(Policy.DEPARTMENT_WRITE)),
):
    name = _normalize_name(payload.name)
    if not name:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_department_name", "message": "Department name cannot be empty"},
        )

    existing = (
        db.query(Departement)
        .filter(func.lower(func.trim(Departement.name)) == name.lower())
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail={"code": "department_already_exists", "message": "Department already exists"},
        )

    departement = Departement(name=name)
    db.add(departement)
    db.commit()
    db.refresh(departement)
    return departement


@router.put("/{department_id}", response_model=DepartementOut)
def update_department(
    department_id: int,
    payload: DepartementUpdate,
    response: Response,
    if_match: Optional[str] = Header(default=None, alias="If-Match"),
    db: Session = Depends(get_db),
    _current_user: User = Depends(require_policy(Policy.DEPARTMENT_WRITE)),
):
    departement = db.get(Departement, department_id)
    if not departement:
        raise HTTPException(
            status_code=404,
            detail={"code": "department_not_found", "message": "Department not found"},
        )
    enforce_if_match(departement, if_match)
    updates = payload.model_dump(exclude_unset=True)
    if "name" in updates and updates["name"] is not None:
        normalized_name = _normalize_name(updates["name"])
        if not normalized_name:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_department_name", "message": "Department name cannot be empty"},
            )

        conflict = (
            db.query(Departement)
            .filter(
                func.lower(func.trim(Departement.name)) == normalized_name.lower(),
                Departement.id != department_id,
            )
            .first()
        )
        if conflict:
            raise HTTPException(
                status_code=409,
                detail={"code": "department_already_exists", "message": "Department already exists"},
            )

        updates["name"] = normalized_name

    for field, value in updates.items():
        setattr(departement, field, value)

    db.commit()
    db.refresh(departement)
    set_etag(response, departement)
    return departement


@router.delete("/{department_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_department(
    department_id: int,
    db: Session = Depends(get_db),
    _current_user: User = Depends(require_policy(Policy.DEPARTMENT_WRITE)),
):
    departement = db.get(Departement, department_id)
    if not departement:
        raise HTTPException(
            status_code=404,
            detail={"code": "department_not_found", "message": "Department not found"},
        )

    db.delete(departement)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
