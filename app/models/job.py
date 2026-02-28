from sqlalchemy import Column, Integer, String, Text, Float, ForeignKey
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
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("job_posts.id"), nullable=False)
    skill_id = Column(Integer, ForeignKey("skills.id"), nullable=False)

    required_level = Column(Integer)  # 1 to 5
    weight = Column(Float)  # importance
    job = relationship("JobPost", back_populates="required_skills")
    skill = relationship("Skill", back_populates="job_requirements")
    
