from sqlalchemy import CheckConstraint, Column, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db.database import Base
from app.models.mixins import AuditMixin


_ENTITY_TYPE_VALUES = "'skill', 'project', 'certification', 'unknown'"
_STATUS_VALUES = "'pending', 'approved', 'rejected'"
_DECISION_VALUES = "'approved', 'rejected'"


class UnknownEntity(AuditMixin, Base):
    __tablename__ = "unknown_entities"
    __table_args__ = (
        UniqueConstraint("normalized_value", name="uq_unknown_entities_normalized_value"),
        CheckConstraint(
            f"entity_type_guess IN ({_ENTITY_TYPE_VALUES})",
            name="ck_unknown_entities_entity_type_guess",
        ),
        CheckConstraint(
            f"resolved_entity_type IS NULL OR resolved_entity_type IN ({_ENTITY_TYPE_VALUES})",
            name="ck_unknown_entities_resolved_entity_type",
        ),
        CheckConstraint(
            f"status IN ({_STATUS_VALUES})",
            name="ck_unknown_entities_status",
        ),
        Index("ix_unknown_entities_status", "status"),
        Index("ix_unknown_entities_candidate_id", "candidate_id"),
        Index("ix_unknown_entities_canonical_skill_id", "canonical_skill_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    raw_value = Column(Text, nullable=False)
    normalized_value = Column(String, nullable=False, index=True)
    entity_type_guess = Column(String, nullable=False, default="unknown", server_default="unknown")
    resolved_entity_type = Column(String, nullable=True)
    status = Column(String, nullable=False, default="pending", server_default="pending")
    source = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    confidence_band = Column(String, nullable=True)
    evidence_json = Column(Text, nullable=False, default="[]", server_default="[]")
    context_excerpt = Column(Text, nullable=True)
    occurrence_count = Column(Integer, nullable=False, default=1, server_default="1")
    candidate_id = Column(Integer, ForeignKey("employees.id"), nullable=True)
    canonical_skill_id = Column(Integer, ForeignKey("skills.id"), nullable=True)

    candidate = relationship("Employee")
    canonical_skill = relationship("Skill")
    reviews = relationship(
        "EntityReview",
        back_populates="unknown_entity",
        cascade="all, delete-orphan",
        order_by="EntityReview.created_at.desc()",
    )


class EntityReview(AuditMixin, Base):
    __tablename__ = "entity_reviews"
    __table_args__ = (
        CheckConstraint(
            f"decision IN ({_DECISION_VALUES})",
            name="ck_entity_reviews_decision",
        ),
        CheckConstraint(
            f"entity_type IN ({_ENTITY_TYPE_VALUES})",
            name="ck_entity_reviews_entity_type",
        ),
        Index("ix_entity_reviews_unknown_entity_id", "unknown_entity_id"),
        Index("ix_entity_reviews_reviewer_id", "reviewer_id"),
        Index("ix_entity_reviews_promoted_skill_id", "promoted_skill_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    unknown_entity_id = Column(Integer, ForeignKey("unknown_entities.id"), nullable=False)
    reviewer_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    decision = Column(String, nullable=False)
    entity_type = Column(String, nullable=False, default="unknown", server_default="unknown")
    canonical_value = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    promoted_skill_id = Column(Integer, ForeignKey("skills.id"), nullable=True)

    unknown_entity = relationship("UnknownEntity", back_populates="reviews")
    reviewer = relationship("User", foreign_keys=[reviewer_id])
    promoted_skill = relationship("Skill")
