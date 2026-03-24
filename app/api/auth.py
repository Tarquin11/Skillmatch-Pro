from typing import Callable
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from app.core.config import settings
from app.db.database import get_db
from app.models.user import User
from app.schemas.user import Token, TokenRefreshRequest, UserCreate, UserResponse
from app.services import auth_service
from app.core.rbac import roles_for_policy

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

@router.post("/signup", response_model=UserResponse)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    return auth_service.create_user(db, user)

@router.post("/login", response_model=Token)
def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    client_ip = request.client.host if request.client else ""
    user_agent = request.headers.get("user-agent", "")
    return auth_service.authenticate_user(
        db,
        form_data.username,
        form_data.password,
        client_ip=client_ip,
        user_agent=user_agent,
    )

@router.post("/refresh", response_model=Token)
def refresh_token(request: Request, payload: TokenRefreshRequest, db: Session = Depends(get_db)):
    client_ip = request.client.host if request.client else ""
    user_agent = request.headers.get("user-agent", "")
    return auth_service.rotate_refresh_token(
        db,
        payload.refresh_token,
        client_ip=client_ip,
        user_agent=user_agent,
    )

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"code": "invalid_credentials", "message": "Incorrect username or password"},
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str | None = payload.get("sub")
        token_version = int(payload.get("tver", 0))
        if email is None:
            raise credentials_exception
    except (JWTError, ValueError):
        raise credentials_exception

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    if token_version != int(user.token_version or 0):
        raise credentials_exception
    return user

def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail={"code": "insufficient_permissions", "message": "Insufficient permissions"})
    return current_user

def require_roles(*allowed_roles: str) -> Callable:
    def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"code": "insufficient_permissions", "message": "Insufficient permissions"},
            )
        return current_user
    return role_checker

@router.get("/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

def require_policy(policy: str) -> Callable:
    allowed_roles = roles_for_policy(policy)

    def policy_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"code": "insufficient_permissions", "message": "Insufficient permissions"},
            )
        return current_user
    return policy_checker
