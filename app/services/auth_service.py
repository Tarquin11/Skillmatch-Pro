from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Deque
from fastapi import HTTPException, status
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.security import create_access_token,generate_refresh_token,get_password_hash,hash_token,verify_password
from app.models.refresh_token import RefreshToken
from app.models.user import User
from app.schemas.user import UserCreate, UserRole

_LOGIN_BUCKETS: dict[str, Deque[float]] = defaultdict(deque)

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _rate_key(email: str, client_ip: str) -> str:
    return f"{email.strip().lower()}|{client_ip or 'unknown'}"

def update_user_role(
    db: Session,
    *,
    actor_user_id: int,
    target_user_id: int,
    new_role: UserRole,
) -> User:
    target_user = db.query(User).filter(User.id == target_user_id).first()
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "user_not_found", "message": "User not found"},
        )

    if target_user.role == new_role:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "role_unchanged", "message": "User already has this role"},
        )

    # Prevent accidental self-lockout of current admin session
    if actor_user_id == target_user.id and target_user.role == "admin" and new_role != "admin":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "self_admin_demotion_forbidden", "message": "Admin cannot demote self"},
        )

    # Ensure at least one admin remains
    if target_user.role == "admin" and new_role != "admin":
        another_admin = db.query(User.id).filter(User.role == "admin", User.id != target_user.id).first()
        if not another_admin:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "last_admin_demotion_forbidden", "message": "At least one admin is required"},
            )

    target_user.role = new_role
    target_user.token_version = int(target_user.token_version or 0) + 1

    db.add(target_user)
    db.commit()
    db.refresh(target_user)
    return target_user

def _another_admin_exists(db: Session, user_id: int) -> bool:
    return db.query(User.id).filter(User.role == "admin", User.id != user_id).first() is not None

def _null_out_nullable_user_references(db: Session, target_user_id: int) -> None:
    inspector = inspect(db.get_bind())
    for table_name in inspector.get_table_names():
        columns = {str(column.get("name")): column for column in inspector.get_columns(table_name)}
        for foreign_key in inspector.get_foreign_keys(table_name):
            if foreign_key.get("referred_table") != "users":
                continue
            referred_columns = foreign_key.get("referred_columns") or []
            constrained_columns = foreign_key.get("constrained_columns") or []
            if "id" not in referred_columns:
                continue
            for column_name in constrained_columns:
                column = columns.get(str(column_name))
                if not column or not column.get("nullable", False):
                    continue
                db.execute(
                    text(
                        f'UPDATE "{table_name}" '
                        f'SET "{column_name}" = NULL '
                        f'WHERE "{column_name}" = :target_user_id'
                    ),
                    {"target_user_id": target_user_id},
                )

def _reassign_required_user_references(
    db: Session,
    *,
    actor_user_id: int,
    target_user_id: int,
) -> None:
    from app.models.active_learning import EntityReview

    db.query(EntityReview).filter(EntityReview.reviewer_id == target_user_id).update(
        {EntityReview.reviewer_id: actor_user_id},
        synchronize_session=False,
    )

def delete_user(
    db: Session,
    *,
    actor_user_id: int,
    target_user_id: int,
) -> None:
    target_user = db.query(User).filter(User.id == target_user_id).first()
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "user_not_found", "message": "User not found"},
        )

    if actor_user_id == target_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "self_user_delete_forbidden", "message": "Admin cannot delete self"},
        )

    if target_user.role == "admin" and not _another_admin_exists(db, target_user.id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "last_admin_delete_forbidden", "message": "At least one admin is required"},
        )

    _null_out_nullable_user_references(db, target_user.id)
    _reassign_required_user_references(
        db,
        actor_user_id=actor_user_id,
        target_user_id=target_user.id,
    )
    db.delete(target_user)
    db.commit()

def _apply_login_rate_limit(email: str, client_ip: str) -> None:
    now_ts = _now_utc().timestamp()
    window = settings.LOGIN_RATE_LIMIT_WINDOW_SECONDS
    max_attempts = settings.LOGIN_RATE_LIMIT_MAX_ATTEMPTS
    key = _rate_key(email, client_ip)
    bucket = _LOGIN_BUCKETS[key]
    while bucket and bucket[0] <= now_ts - window:
        bucket.popleft()
    if len(bucket) >= max_attempts:
        retry_after = max(1, int(window - (now_ts - bucket[0])))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "login_rate_limited",
                "message": "Too many login attempts, try again later",
                "retry_after_seconds": retry_after,
            },
            headers={"Retry-After": str(retry_after)},
        )
    bucket.append(now_ts)

def _enforce_lockout(user: User, now: datetime) -> None:
    locked_until = _as_utc(user.locked_until)
    if locked_until and locked_until > now:
        retry_after = max(1, int((locked_until - now).total_seconds()))
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail={
                "code": "account_locked",
                "message": "Account is temporarily locked",
                "retry_after_seconds": retry_after,
            },
            headers={"Retry-After": str(retry_after)},
        )
    
def _record_failed_login(db: Session, user: User, now: datetime) -> None:
    user.failed_login_attempts = int(user.failed_login_attempts or 0) + 1
    user.last_failed_login_at = now

    if user.failed_login_attempts >= settings.LOGIN_MAX_FAILED_ATTEMPTS:
        streak = user.failed_login_attempts - settings.LOGIN_MAX_FAILED_ATTEMPTS
        lock_seconds = min(
            settings.LOGIN_LOCKOUT_MAX_SECONDS,
            settings.LOGIN_LOCKOUT_BASE_SECONDS * (2 ** max(0, streak)),
        )
        user.locked_until = now + timedelta(seconds=lock_seconds)
    db.add(user)
    db.commit()

def _reset_failed_login_state(user: User) -> None:
    user.failed_login_attempts = 0
    user.last_failed_login_at = None
    user.locked_until = None

def create_user(
    db: Session,
    user: UserCreate,
    *,
    allow_admin_role: bool = False,
    actor_user_id: int | None = None,
):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail={"code": "email_already_registered", "message": "Email already registered"})
    is_first_user = db.query(User.id).first() is None
    requested_role = user.role
    if is_first_user:
        role = requested_role if requested_role is not None else "admin"
    else:
        if requested_role == "admin" and not allow_admin_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "admin_signup_forbidden",
                    "message": "Admin role cannot be self-assigned",
                },
            )
        role = requested_role if requested_role is not None else "user"
    hashed_pwd = get_password_hash(user.password)
    new_user = User(
        email=user.email,
        hashed_password=hashed_pwd,
        role=role,
        created_by=actor_user_id,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

def authenticate_user(db: Session, email: str, password: str, client_ip: str = "", user_agent: str = ""):
    _apply_login_rate_limit(email, client_ip)

    invalid_credentials = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"code": "invalid_credentials", "message": "Incorrect username or password"},
        headers={"WWW-Authenticate": "Bearer"},
    )
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise invalid_credentials
    now = _now_utc()
    _enforce_lockout(user, now)
    if not verify_password(password, user.hashed_password):
        _record_failed_login(db, user, now)
        raise invalid_credentials
    _reset_failed_login_state(user)
    db.add(user)
    access_ttl = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=access_ttl,
        token_version=int(user.token_version or 0),
    )
    refresh_token = generate_refresh_token()
    refresh_hash = hash_token(refresh_token)
    refresh_expires_at = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    db.add(
        RefreshToken(
            user_id=user.id,
            token_hash=refresh_hash,
            expires_at=refresh_expires_at,
            ip_address=(client_ip or None),
            user_agent=(user_agent or None),
        )
    )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "account_inactive", "message": "Account is inactive"},
    )

    db.commit()
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": int(access_ttl.total_seconds()),
        "refresh_token": refresh_token,
    }

def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def rotate_refresh_token(db: Session, refresh_token: str, client_ip: str = "", user_agent: str = ""):
    now = _now_utc()
    raw_hash = hash_token(refresh_token)

    record = db.query(RefreshToken).filter(RefreshToken.token_hash == raw_hash).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "invalid_refresh_token", "message": "Invalid refresh token"},
        )
    user = db.query(User).filter(User.id == record.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "invalid_refresh_token", "message": "Invalid refresh token"},
        )
    if record.revoked_at is not None:
        user.token_version = int(user.token_version or 0) + 1
        db.query(RefreshToken).filter(
            RefreshToken.user_id == user.id,
            RefreshToken.revoked_at.is_(None),
        ).update({"revoked_at": now}, synchronize_session=False)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "refresh_token_reuse_detected", "message": "Refresh token reuse detected, please login again"},
        )
    expires_at = _as_utc(record.expires_at)
    if expires_at is None or expires_at <= now:
        record.revoked_at = now
        db.add(record)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "expired_refresh_token", "message": "Refresh token expired"},
    )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "insufficient_permissions", "message": "Insufficient permissions"},
        )
    new_refresh_token = generate_refresh_token()
    new_refresh_hash = hash_token(new_refresh_token)
    record.revoked_at = now
    record.replaced_by_token_hash = new_refresh_hash
    db.add(record)
    db.add(
        RefreshToken(
            user_id=user.id,
            token_hash=new_refresh_hash,
            expires_at=now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
            ip_address=(client_ip or None),
            user_agent=(user_agent or None),
        )
    )
    access_ttl = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=access_ttl,
        token_version=int(user.token_version or 0),
    )
    db.commit()
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": int(access_ttl.total_seconds()),
        "refresh_token": new_refresh_token,
    }
