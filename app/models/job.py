from sqlalchemy import Column, Integer, String, Text, Float, ForeignKey, CheckConstraint,Index, UniqueConstraint
from sqlalchemy.orm import relationship, synonym
from app.db.database import Base

class JobPost(Base):
    __tablename__ = "job_posts"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text)
    department = Column("departement", String, index=True)
    required_skills = relationship("JobSkill", back_populates="job")
    departement = synonym("department")

class JobSkill(Base):
    __tablename__ = "job_skills"
    __table_args__ = (
        UniqueConstraint("job_id", "skill_id", name="uq_job_skills_job_skill"),
        CheckConstraint(
            "required_level IS NULL OR (required_level BETWEEN 1 AND 5)",
            name="ck_job_skills_required_level_range",
        ),
        CheckConstraint(
            "weight IS NULL OR weight > 0",
            name="ck_job_skills_weight_positive",
        ),
        Index("ix_job_skills_job_id", "job_id"),
        Index("ix_job_skills_skill_id", "skill_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("job_posts.id"), nullable=False)
    skill_id = Column(Integer, ForeignKey("skills.id"), nullable=False)
    required_level = Column(Integer)  # 1 to 5
    weight = Column(Float)  # importance
    job = relationship("JobPost", back_populates="required_skills")
    skill = relationship("Skill", back_populates="job_requirements")
    
