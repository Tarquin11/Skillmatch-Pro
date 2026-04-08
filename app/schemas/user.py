from typing import Literal
from pydantic import EmailStr, ConfigDict
from app.schemas.common import StrictBaseModel

UserRole = Literal["admin", "recruiter", "user"]

class UserCreate(StrictBaseModel):
    email: EmailStr
    password: str
    role: UserRole | None = None

class UserResponse(StrictBaseModel):
    id: int
    # Keep response tolerant for legacy seeded records (e.g. *.local),
    # while UserCreate remains strict with EmailStr validation.
    email: str
    is_active: bool
    role: UserRole
    model_config = ConfigDict(from_attributes=True, extra="forbid")

class Token(StrictBaseModel):
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: str

class TokenRefreshRequest(StrictBaseModel):
    refresh_token: str

class UserRoleUpdateRequest(StrictBaseModel):
    role: UserRole
