from typing import List
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.job import JobPost, JobSkill
from app.models.skill import Skill
from app.schemas.job import JobCreate, JobOut, JobSkillRequirementOut, JobSkillRequirementRequest, JobUpdate
from app.api.auth import get_current_active_user, require_roles
from app.models.user import User

router = APIRouter(prefix="/jobs", tags=["jobs"], dependencies=[Depends(get_current_active_user)])

@router.get("/", response_model=List[JobOut])
def list_jobs(db: Session = Depends(get_db)):
    return db.query(JobPost).order_by(JobPost.id.desc()).all()


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(JobPost, job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "Job not found"})
    return job


@router.post("/", response_model=JobOut, status_code=status.HTTP_201_CREATED)
def create_job(payload: JobCreate, db: Session = Depends(get_db), _current_user: User = Depends(require_roles("admin"))):
    job = JobPost(**payload.model_dump(exclude_unset=True, by_alias=False))
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


@router.put("/{job_id}", response_model=JobOut)
def update_job(job_id: int, payload: JobUpdate, db: Session = Depends(get_db), _current_user: User = Depends(require_roles("admin"))):
    job = db.get(JobPost, job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "Job not found"})

    updates = payload.model_dump(exclude_unset=True, by_alias=False)
    for field, value in updates.items():
        setattr(job, field, value)

    db.commit()
    db.refresh(job)
    return job


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_job(job_id: int, db: Session = Depends(get_db), _current_user: User = Depends(require_roles("admin"))):
    job = db.get(JobPost, job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "Job not found"})

    db.delete(job)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{job_id}/skills", response_model=List[JobSkillRequirementOut])
def list_job_skill_requirements(job_id: int, db: Session = Depends(get_db)):
    job = db.get(JobPost, job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "Job not found"})

    return (
        db.query(JobSkill)
        .filter(JobSkill.job_id == job_id)
        .order_by(JobSkill.id.asc())
        .all()
    )


@router.post("/{job_id}/skills", response_model=JobSkillRequirementOut)
def upsert_job_skill_requirement(job_id: int,payload: JobSkillRequirementRequest,db: Session = Depends(get_db),_current_user: User = Depends(require_roles("admin"))):
    job = db.get(JobPost, job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "Job not found"})

    skill = db.get(Skill, payload.skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail={"code": "skill_not_found", "message": "Skill not found"})

    requirement = (
        db.query(JobSkill)
        .filter(
            JobSkill.job_id == job_id,
            JobSkill.skill_id == payload.skill_id,
        )
        .first()
    )

    if requirement:
        requirement.required_level = payload.required_level
        requirement.weight = payload.weight
    else:
        requirement = JobSkill(
            job_id=job_id,
            skill_id=payload.skill_id,
            required_level=payload.required_level,
            weight=payload.weight,
        )
        db.add(requirement)

    db.commit()
    db.refresh(requirement)
    return requirement


@router.delete(
    "/{job_id}/skills/{skill_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def remove_job_skill_requirement(job_id: int, skill_id: int, db: Session = Depends(get_db), _current_user: User = Depends(require_roles("admin"))):
    requirement = (
        db.query(JobSkill)
        .filter(
            JobSkill.job_id == job_id,
            JobSkill.skill_id == skill_id,
        )
        .first()
    )
    if not requirement:
        raise HTTPException(status_code=404, detail={"code": "job_skill_requirement_not_found", "message": "Job-skill requirement not found"})

    db.delete(requirement)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)