from datetime import date
from typing import Optional
from pydantic import BaseModel, Field , EmailStr

class EmployeeBase(BaseModel):
    employee_number: Optional[str] = Field(default=None, alias="employeeNumber")
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    full_name: Optional[str] = None
    email: Optional[str] = None
    department: Optional[str] = Field(default=None, alias="departement")
    position: Optional[str] = None
    performance_score: Optional[str] = None
    hire_date : Optional[date] = None

    class Config:
        from_attributes = True
        populate_by_name = True

class EmployeeCreate(EmployeeBase):
    employee_number: str = Field(alias="employeeNumber")  
    first_name: str
    last_name: str
    email: EmailStr

class EmployeeUpdate(EmployeeBase):
    email: Optional[EmailStr] = None


class EmployeeOut(EmployeeBase):
    id: int