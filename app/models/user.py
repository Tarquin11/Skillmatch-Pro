from sqlalchemy import Boolean, Column, Integer, String, Index
from app.db.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    role = Column(String, default="user", nullable=False)

    __tableargs__ = (
        Index("ix_users_role", role),
    )