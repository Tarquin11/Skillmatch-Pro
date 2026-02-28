import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.security import get_password_hash
from app.db.database import SessionLocal
from app.models.Employee_skill import EmployeeSkill
from app.models.employee import Employee
from app.models.job import JobPost, JobSkill
from app.models.skill import Skill
from app.models.user import User

logger = logging.getLogger(__name__)


@dataclass
class SeedStats:
    users_created: int = 0
    users_updated: int = 0
    skills_created: int = 0
    skills_existing: int = 0
    employees_created: int = 0
    employees_updated: int = 0
    employee_skills_created: int = 0
    employee_skills_updated: int = 0
    jobs_created: int = 0
    jobs_updated: int = 0
    job_requirements_created: int = 0
    job_requirements_updated: int = 0


def upsert_user(db: Session, email: str, password: str, role: str, stats: SeedStats):
    user = db.query(User).filter(User.email == email).first()
    if not user:
        payload = {
            "email": email,
            "hashed_password": get_password_hash(password),
            "is_active": True,
        }
        if hasattr(User, "role"):
            payload["role"] = role
        db.add(User(**payload))
        stats.users_created += 1
        return

    changed = False
    if not user.is_active:
        user.is_active = True
        changed = True

    if hasattr(user, "role") and user.role != role:
        user.role = role
        changed = True

    if changed:
        stats.users_updated += 1


def get_or_create_skill(db: Session, name: str, stats: SeedStats) -> Skill:
    skill = db.query(Skill).filter(Skill.name == name).first()
    if skill:
        stats.skills_existing += 1
        return skill
    skill = Skill(name=name)
    db.add(skill)
    db.flush()
    stats.skills_created += 1
    return skill


def upsert_employee(db: Session, payload: dict, stats: SeedStats) -> Employee:
    employee = (
        db.query(Employee)
        .filter(Employee.employee_number == payload["employee_number"])
        .first()
    )
    if not employee:
        employee = Employee(**payload)
        db.add(employee)
        db.flush()
        stats.employees_created += 1
        return employee

    changed = False
    for field, value in payload.items():
        if value is None:
            continue
        if getattr(employee, field) != value:
            setattr(employee, field, value)
            changed = True

    if changed:
        stats.employees_updated += 1
    return employee


def upsert_employee_skill(
    db: Session, employee_id: int, skill_id: int, level: int, stats: SeedStats
):
    link = (
        db.query(EmployeeSkill)
        .filter(
            EmployeeSkill.employee_id == employee_id,
            EmployeeSkill.skill_id == skill_id,
        )
        .first()
    )
    if not link:
        db.add(EmployeeSkill(employee_id=employee_id, skill_id=skill_id, level=level))
        stats.employee_skills_created += 1
        return

    if link.level != level:
        link.level = level
        stats.employee_skills_updated += 1


def upsert_job(db: Session, title: str, description: str, department: str, stats: SeedStats) -> JobPost:
    job = (
        db.query(JobPost)
        .filter(JobPost.title == title, JobPost.department == department)
        .first()
    )
    if not job:
        job = JobPost(title=title, description=description, department=department)
        db.add(job)
        db.flush()
        stats.jobs_created += 1
        return job

    changed = False
    if job.description != description:
        job.description = description
        changed = True
    if changed:
        stats.jobs_updated += 1
    return job


def upsert_job_requirement(
    db: Session, job_id: int, skill_id: int, required_level: int, weight: float, stats: SeedStats
):
    req = (
        db.query(JobSkill)
        .filter(JobSkill.job_id == job_id, JobSkill.skill_id == skill_id)
        .first()
    )
    if not req:
        db.add(
            JobSkill(
                job_id=job_id,
                skill_id=skill_id,
                required_level=required_level,
                weight=weight,
            )
        )
        stats.job_requirements_created += 1
        return

    changed = False
    if req.required_level != required_level:
        req.required_level = required_level
        changed = True
    if req.weight != weight:
        req.weight = weight
        changed = True
    if changed:
        stats.job_requirements_updated += 1


def seed_demo_data() -> SeedStats:
    stats = SeedStats()
    db = SessionLocal()

    try:
        # Users
        upsert_user(db, "admin@skillmatch.local", "Admin123!", "admin", stats)
        upsert_user(db, "user@skillmatch.local", "User123!", "user", stats)

        # Skills
        skill_names = [
            "Python",
            "FastAPI",
            "SQL",
            "Angular",
            "Machine Learning",
            "Data Analysis",
            "Docker",
        ]
        skills = {name: get_or_create_skill(db, name, stats) for name in skill_names}

        # Employees
        e1 = upsert_employee(
            db,
            {
                "employee_number": "E-1001",
                "first_name": "Sara",
                "last_name": "Ben Ali",
                "full_name": "Sara Ben Ali",
                "email": "sara.benali@company.local",
                "department": "Engineering",
                "position": "Backend Developer",
                "performance_score": "superior",
                "employment_status": "active",
            },
            stats,
        )
        e2 = upsert_employee(
            db,
            {
                "employee_number": "E-1002",
                "first_name": "Youssef",
                "last_name": "Trabelsi",
                "full_name": "Youssef Trabelsi",
                "email": "youssef.trabelsi@company.local",
                "department": "Data",
                "position": "Data Analyst",
                "performance_score": "acceptable",
                "employment_status": "active",
            },
            stats,
        )

        # Employee skills
        upsert_employee_skill(db, e1.id, skills["Python"].id, 5, stats)
        upsert_employee_skill(db, e1.id, skills["FastAPI"].id, 4, stats)
        upsert_employee_skill(db, e1.id, skills["SQL"].id, 4, stats)
        upsert_employee_skill(db, e2.id, skills["Python"].id, 3, stats)
        upsert_employee_skill(db, e2.id, skills["Data Analysis"].id, 5, stats)
        upsert_employee_skill(db, e2.id, skills["Machine Learning"].id, 3, stats)

        # Jobs
        j1 = upsert_job(
            db,
            title="Senior Backend Engineer",
            description="Build and maintain internal FastAPI services.",
            department="Engineering",
            stats=stats,
        )
        j2 = upsert_job(
            db,
            title="ML Analyst",
            description="Analyze workforce data and assist matching models.",
            department="Data",
            stats=stats,
        )

        # Job requirements
        upsert_job_requirement(db, j1.id, skills["Python"].id, 4, 1.5, stats)
        upsert_job_requirement(db, j1.id, skills["FastAPI"].id, 4, 1.2, stats)
        upsert_job_requirement(db, j1.id, skills["SQL"].id, 3, 1.0, stats)

        upsert_job_requirement(db, j2.id, skills["Data Analysis"].id, 4, 1.4, stats)
        upsert_job_requirement(db, j2.id, skills["Machine Learning"].id, 3, 1.2, stats)
        upsert_job_requirement(db, j2.id, skills["Python"].id, 3, 1.0, stats)

        db.commit()

        logger.info(
            "Seed summary | users(created=%d updated=%d) skills(created=%d existing=%d) "
            "employees(created=%d updated=%d) employee_skills(created=%d updated=%d) "
            "jobs(created=%d updated=%d) job_requirements(created=%d updated=%d)",
            stats.users_created,
            stats.users_updated,
            stats.skills_created,
            stats.skills_existing,
            stats.employees_created,
            stats.employees_updated,
            stats.employee_skills_created,
            stats.employee_skills_updated,
            stats.jobs_created,
            stats.jobs_updated,
            stats.job_requirements_created,
            stats.job_requirements_updated,
        )
        return stats
    except Exception:
        db.rollback()
        logger.exception("Seed failed; transaction rolled back.")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    seed_demo_data()
