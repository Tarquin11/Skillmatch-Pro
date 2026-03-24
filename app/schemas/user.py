from typing import Literal
from pydantic import EmailStr, ConfigDict
from app.schemas.common import StrictBaseModel

UserRole = Literal["admin", "user"]

class UserCreate(StrictBaseModel):
    email: EmailStr
    password: str
    role: UserRole | None = None

class UserResponse(StrictBaseModel):
    id: int
    email: EmailStr
    is_active: bool
    role: UserRole
    model_config = ConfigDict(from_attributes=True, extra="forbid")

class Token(StrictBaseModel):
    access_token: str
    token_type: str
    
