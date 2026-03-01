from typing import Literal
from pydantic import BaseModel, EmailStr, ConfigDict

UserRole = Literal["admin", "user"]

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    role: UserRole | None = None

class UserResponse(BaseModel):
    id: int
    email: EmailStr
    is_active: bool
    role: UserRole
    model_config = ConfigDict(from_attributes=True)

class Token(BaseModel):
    access_token: str
    token_type: str
    
