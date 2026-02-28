from typing import List 
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.skill import Skill
from app.models.employee import Employee
from app.models.Employee_skill import EmployeeSkill
from app.schemas.skill import SkillCreate, SkillOut, SkillUpdate, EmployeeSkillOut, EmployeeSkillAssignRequest
from app.api.auth import get_current_active_user, require_roles
from app.models.user import User

router = APIRouter(prefix="/skills", tags=["skills"], dependencies=[Depends(get_current_active_user)])

@router.get("/", response_model=List[SkillOut])
def list_skills(db: Session = Depends(get_db)):
    return db.query(Skill).order_by(Skill.name.asc()).all()

@router.get("/{skill_id}", response_model=SkillOut)
def get_skill(skill_id: int, db: Session = Depends(get_db)):
    skill = db.get(Skill, skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail={"code": "skill_not_found", "message": "Skill not found / Compétance pas trouvé "})
    return skill


@router.post("/", response_model=SkillOut, status_code=status.HTTP_201_CREATED)
def create_skill(payload: SkillCreate, db: Session = Depends(get_db), _current_user: User = Depends(require_roles("admin"))):
    existing = db.query(Skill).filter(Skill.name == payload.name).first()
    if existing:
        raise HTTPException(status_code=409, detail={"code": "skill_already_exists", "message": "Skill already exists"})

    skill = Skill(name=payload.name)
    db.add(skill)
    db.commit()
    db.refresh(skill)
    return skill


@router.put("/{skill_id}", response_model=SkillOut)
def update_skill(skill_id: int, payload: SkillUpdate, db: Session = Depends(get_db), _current_user: User = Depends(require_roles("admin"))):
    skill = db.get(Skill, skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail={"code": "skill_not_found", "message": "Skill not found / Compétance pas trouvé "})

    updates = payload.model_dump(exclude_unset=True)
    for field, value in updates.items():
        setattr(skill, field, value)

    db.commit()
    db.refresh(skill)
    return skill


@router.delete("/{skill_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_skill(skill_id: int, db: Session = Depends(get_db), _current_user: User = Depends(require_roles("admin"))):
    skill = db.get(Skill, skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail={"code": "skill_not_found", "message": "Skill not found / Compétance pas trouvé "})

    db.delete(skill)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/employees/{employee_id}", response_model=EmployeeSkillOut)
def assign_skill_to_employee(employee_id: int,payload: EmployeeSkillAssignRequest,db: Session = Depends(get_db),_current_user: User = Depends(require_roles("admin"))):
    employee = db.get(Employee, employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail={"code": "employee_not_found", "message": "Employee not found"})

    skill = db.get(Skill, payload.skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail={"code": "skill_not_found", "message": "Skill not found / Compétance pas trouvé "})

    link = (
        db.query(EmployeeSkill)
        .filter(
            EmployeeSkill.employee_id == employee_id,
            EmployeeSkill.skill_id == payload.skill_id,
        )
        .first()
    )
    if link:
        link.level = payload.level
    else:
        link = EmployeeSkill(
            employee_id=employee_id,
            skill_id=payload.skill_id,
            level=payload.level,
        )
        db.add(link)

    db.commit()
    db.refresh(link)
    return link


@router.delete(
    "/employees/{employee_id}/{skill_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def unassign_skill_from_employee(employee_id: int,skill_id: int,db: Session = Depends(get_db), _current_user: User = Depends(require_roles("admin"))):
    link = (
        db.query(EmployeeSkill)
        .filter(
            EmployeeSkill.employee_id == employee_id,
            EmployeeSkill.skill_id == skill_id,
        )
        .first()
    )
    if not link:
        raise HTTPException(status_code=404, detail={"code": "skill_not_found", "message": "Skill not found / Compétance pas trouvé "})

    db.delete(link)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)