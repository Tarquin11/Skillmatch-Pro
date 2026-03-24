from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship, synonym
from app.db.database import Base
from app.models.mixins import AuditMixin

class Skill(AuditMixin, Base):
    __tablename__ = "skills"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    employee_skills = relationship(
        "EmployeeSkill",
        back_populates="skill",
        cascade="all, delete-orphan",
    )
    job_requirements = relationship(
        "JobSkill",
        back_populates="skill",
        cascade="all, delete-orphan",
    )

    # Backward compatibility alias for existing code.
    employees = synonym("employee_skills")
