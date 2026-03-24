from sqlalchemy import Column, DateTime, ForeignKey, Integer, func


class AuditMixin:
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        index=True,
    )
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)

