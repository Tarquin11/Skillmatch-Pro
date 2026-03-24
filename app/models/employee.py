from sqlalchemy import Column, Date, Float, Index, Integer, String
from sqlalchemy.orm import relationship, synonym
from app.db.database import Base

class Employee(Base):
    __tablename__ = "employees"

    # identifients primaires et basiques
    id = Column(Integer, primary_key=True, index=True)
    employee_number = Column("Employee_number", String, unique=True, index=True, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    full_name = Column(String, nullable=False)

    # Demographiques
    email = Column(String, unique=True, index=True, nullable=False)
    gender = Column(String, nullable=True)
    dob = Column(Date, nullable=True)
    marital_status = Column(String, nullable=True)
    citizenship = Column(String, nullable=True)

    # Location
    governorate = Column("Governorate", String, nullable=True)
    city = Column(String, nullable=True)
    zip_code = Column(String, nullable=True)

    # details professionnels
    department = Column("departement", String, nullable=True)
    position = Column(String, nullable=True)
    manager_name = Column(String, nullable=True)
    employment_status = Column("Employment_status", String, default="active")

    # Finance et performance
    salary = Column(Float, nullable=True)
    pay_rate = Column(Float, nullable=True)
    performance_score = Column(String, nullable=True)
    engagement_survey = Column(Float, nullable=True)
    emp_satisfaction = Column(Integer, nullable=True)

    # Dates d'emploi
    hire_date = Column(Date, nullable=True)
    termination_date = Column(Date, nullable=True)
    termination_reason = Column(String, nullable=True)

    # Recrutement
    recruitment_source = Column("recruitement_source", String, nullable=True)
    skills = relationship("EmployeeSkill", back_populates="employee", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_employees_departement", department),
        Index("ix_employees_position", position),
    )

    # compagnie de synonymes pour les champs avec des noms differents dans la base de donnees
    Employee_number = synonym("employee_number")
    Governorate = synonym("governorate")
    departement = synonym("department")
    Employment_status = synonym("employment_status")
    recruitement_source = synonym("recruitment_source")
