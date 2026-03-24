from sqlalchemy import CheckConstraint, Column, ForeignKey, Index, Integer, UniqueConstraint
from sqlalchemy.orm import relationship
from app.db.database import Base
from app.models.mixins import AuditMixin

class EmployeeSkill(AuditMixin, Base):
    __tablename__ = "employee_skills"
    __table_args__ = (
        UniqueConstraint("employee_id", "skill_id", name="uq_employee_skills_employee_skill"),
        CheckConstraint("level IS NULL OR (level BETWEEN 1 AND 5)", name="ck_employee_skills_level_range"),
        Index("ix_employee_skills_employee_id", "employee_id"),
        Index("ix_employee_skills_skill_id", "skill_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    skill_id = Column(Integer, ForeignKey("skills.id"), nullable=False)
    level = Column(Integer)  # 1 to 5
    employee = relationship("Employee", back_populates="skills")
    skill = relationship("Skill", back_populates="employee_skills")
