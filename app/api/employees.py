from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status, Header
from sqlalchemy import func, or_
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.employee import Employee
from app.schemas.employee import EmployeeCreate, EmployeeOut, EmployeeUpdate
from app.api.auth import get_current_active_user, require_policy
from app.models.user import User
from app.api.utils import apply_list_query
from app.schemas.listing import ListQuery
from app.api.concurrency import enforce_if_match, set_etag
from app.core.rbac import Policy

router = APIRouter(dependencies=[Depends(get_current_active_user)])

_EMPLOYEE_SORT_FIELDS = {
    "id": Employee.id,
    "first_name": Employee.first_name,
    "last_name": Employee.last_name,
    "email": Employee.email,
    "department": Employee.department,
    "position": Employee.position,
    "hire_date": Employee.hire_date,
}
@router.get("/", response_model=List[EmployeeOut])
def read_employees(
    params: ListQuery = Depends(),
    department: Optional[str] = Query(default=None),
    position: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    query = db.query(Employee)
    if params.search:
        term = params.search.strip().lower()
        query = query.filter(
            or_(
                func.lower(func.trim(Employee.first_name)).like(f"%{term}%"),
                func.lower(func.trim(Employee.last_name)).like(f"%{term}%"),
                func.lower(func.trim(Employee.email)).like(f"%{term}%"),
                func.lower(func.trim(Employee.position)).like(f"%{term}%"),
                func.lower(func.trim(Employee.department)).like(f"%{term}%"),
            )
        )
    if department:
        dep = department.strip()
        query = query.filter(func.lower(func.trim(Employee.department)).like(f"%{dep.lower()}%"))
    if position:
        pos = position.strip()
        query = query.filter(func.lower(func.trim(Employee.position)).like(f"%{pos.lower()}%"))

    return apply_list_query(
        query,
        sort_by=params.sort_by,
        sort_dir=params.sort_dir,
        sort_map=_EMPLOYEE_SORT_FIELDS,
        skip=params.skip,
        limit=params.limit,
    ).all()
@router.get("/{employee_id}", response_model=EmployeeOut)
def get_employee(employee_id: int, response: Response, db: Session = Depends(get_db)):
    employee = db.get(Employee, employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail={"code": "employee_not_found", "message": "Employee not found"})
    set_etag(response, employee)
    return employee

@router.post("/", response_model=EmployeeOut, status_code=status.HTTP_201_CREATED)
def create_employee(payload: EmployeeCreate, db: Session = Depends(get_db), _current_user: User = Depends(require_policy(Policy.EMPLOYEE_WRITE))):
    data = payload.model_dump(by_alias=False)

    existing = db.query(Employee).filter(Employee.employee_number == data["employee_number"]).first()
    if existing:
        raise HTTPException(status_code=409, detail={"code": "employee_number_already_exists", "message": "Employee number already exists"})

    existing_email = db.query(Employee).filter(Employee.email == data["email"]).first()
    if existing_email:
        raise HTTPException(status_code=409, detail={"code": "email_already_registered", "message": "Email already registered"})

    employee = Employee(**data)
    db.add(employee)
    db.commit()
    db.refresh(employee)
    return employee


@router.put("/{employee_id}", response_model=EmployeeOut)
def update_employee(employee_id: int, payload: EmployeeUpdate,response: Response,if_match: Optional[str] = Header(default=None, alias="If-Match"), db: Session = Depends(get_db), _current_user: User = Depends(require_policy(Policy.EMPLOYEE_WRITE))):
    employee = db.get(Employee, employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail={"code": "employee_not_found", "message": "Employee not found / Employée Pas trouvé"})

    enforce_if_match(employee, if_match)
    updates = payload.model_dump(exclude_unset=True, by_alias=False)
    for field, value in updates.items():
        setattr(employee, field, value)

    db.commit()
    db.refresh(employee)
    set_etag(response, employee)
    return employee


@router.delete("/{employee_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_employee(employee_id: int, db: Session = Depends(get_db), _current_user: User = Depends(require_policy(Policy.EMPLOYEE_WRITE))):
    employee = db.get(Employee, employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail={"code": "employee_not_found", "message": "Employee not found / Employée Pas trouvé"})

    db.delete(employee)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
