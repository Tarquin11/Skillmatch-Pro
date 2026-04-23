from datetime import datetime
from typing import Literal

from pydantic import ConfigDict, Field

from app.schemas.common import StrictBaseModel


EntityType = Literal["skill", "project", "certification", "unknown"]
ReviewDecision = Literal["approved", "rejected"]
ReviewStatus = Literal["pending", "approved", "rejected"]


class EntityReviewCreate(StrictBaseModel):
    decision: ReviewDecision
    entity_type: EntityType = "unknown"
    canonical_value: str | None = None
    notes: str | None = None


class EntityReviewOut(StrictBaseModel):
    id: int
    unknown_entity_id: int
    reviewer_id: int
    decision: ReviewDecision
    entity_type: EntityType
    canonical_value: str | None = None
    notes: str | None = None
    promoted_skill_id: int | None = None
    created_at: datetime | None = None
    model_config = ConfigDict(from_attributes=True, extra="forbid")


class UnknownEntityOut(StrictBaseModel):
    id: int
    raw_value: str
    normalized_value: str
    entity_type_guess: EntityType
    resolved_entity_type: EntityType | None = None
    status: ReviewStatus
    source: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_band: str | None = None
    evidence: list[str] = Field(default_factory=list)
    context_excerpt: str | None = None
    occurrence_count: int = Field(default=1, ge=1)
    candidate_id: int | None = None
    canonical_skill_id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    reviews: list[EntityReviewOut] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")
