from sqlalchemy import Boolean, Column, Index, Integer, String
from app.db.database import Base
from app.models.mixins import AuditMixin


class User(AuditMixin, Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    role = Column(String, default="user", nullable=False)

    __table_args__ = (
        Index("ix_users_role", role),
    )
