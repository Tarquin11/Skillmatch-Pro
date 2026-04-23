"""add active learning tables

Revision ID: 0009_add_active_learning_tables
Revises: 0008_add_candidate_boards_to_employees
Create Date: 2026-04-23
"""

from alembic import op
import sqlalchemy as sa


revision = "0009_add_active_learning_tables"
down_revision = "0008_add_candidate_boards_to_employees"
branch_labels = None
depends_on = None


ENTITY_TYPE_VALUES = "'skill', 'project', 'certification', 'unknown'"
STATUS_VALUES = "'pending', 'approved', 'rejected'"
DECISION_VALUES = "'approved', 'rejected'"


def upgrade() -> None:
    op.create_table(
        "unknown_entities",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("raw_value", sa.Text(), nullable=False),
        sa.Column("normalized_value", sa.String(), nullable=False),
        sa.Column("entity_type_guess", sa.String(), nullable=False, server_default="unknown"),
        sa.Column("resolved_entity_type", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("source", sa.String(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("confidence_band", sa.String(), nullable=True),
        sa.Column("evidence_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("context_excerpt", sa.Text(), nullable=True),
        sa.Column("occurrence_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("candidate_id", sa.Integer(), nullable=True),
        sa.Column("canonical_skill_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("created_by", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            f"entity_type_guess IN ({ENTITY_TYPE_VALUES})",
            name="ck_unknown_entities_entity_type_guess",
        ),
        sa.CheckConstraint(
            f"resolved_entity_type IS NULL OR resolved_entity_type IN ({ENTITY_TYPE_VALUES})",
            name="ck_unknown_entities_resolved_entity_type",
        ),
        sa.CheckConstraint(
            f"status IN ({STATUS_VALUES})",
            name="ck_unknown_entities_status",
        ),
        sa.ForeignKeyConstraint(["candidate_id"], ["employees.id"]),
        sa.ForeignKeyConstraint(["canonical_skill_id"], ["skills.id"]),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("normalized_value", name="uq_unknown_entities_normalized_value"),
    )
    op.create_index("ix_unknown_entities_id", "unknown_entities", ["id"], unique=False)
    op.create_index("ix_unknown_entities_normalized_value", "unknown_entities", ["normalized_value"], unique=False)
    op.create_index("ix_unknown_entities_status", "unknown_entities", ["status"], unique=False)
    op.create_index("ix_unknown_entities_candidate_id", "unknown_entities", ["candidate_id"], unique=False)
    op.create_index("ix_unknown_entities_canonical_skill_id", "unknown_entities", ["canonical_skill_id"], unique=False)
    op.create_index("ix_unknown_entities_created_at", "unknown_entities", ["created_at"], unique=False)
    op.create_index("ix_unknown_entities_updated_at", "unknown_entities", ["updated_at"], unique=False)
    op.create_index("ix_unknown_entities_created_by", "unknown_entities", ["created_by"], unique=False)

    op.create_table(
        "entity_reviews",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("unknown_entity_id", sa.Integer(), nullable=False),
        sa.Column("reviewer_id", sa.Integer(), nullable=False),
        sa.Column("decision", sa.String(), nullable=False),
        sa.Column("entity_type", sa.String(), nullable=False, server_default="unknown"),
        sa.Column("canonical_value", sa.String(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("promoted_skill_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("created_by", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            f"decision IN ({DECISION_VALUES})",
            name="ck_entity_reviews_decision",
        ),
        sa.CheckConstraint(
            f"entity_type IN ({ENTITY_TYPE_VALUES})",
            name="ck_entity_reviews_entity_type",
        ),
        sa.ForeignKeyConstraint(["unknown_entity_id"], ["unknown_entities.id"]),
        sa.ForeignKeyConstraint(["reviewer_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["promoted_skill_id"], ["skills.id"]),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_entity_reviews_id", "entity_reviews", ["id"], unique=False)
    op.create_index("ix_entity_reviews_unknown_entity_id", "entity_reviews", ["unknown_entity_id"], unique=False)
    op.create_index("ix_entity_reviews_reviewer_id", "entity_reviews", ["reviewer_id"], unique=False)
    op.create_index("ix_entity_reviews_promoted_skill_id", "entity_reviews", ["promoted_skill_id"], unique=False)
    op.create_index("ix_entity_reviews_created_at", "entity_reviews", ["created_at"], unique=False)
    op.create_index("ix_entity_reviews_updated_at", "entity_reviews", ["updated_at"], unique=False)
    op.create_index("ix_entity_reviews_created_by", "entity_reviews", ["created_by"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_entity_reviews_created_by", table_name="entity_reviews")
    op.drop_index("ix_entity_reviews_updated_at", table_name="entity_reviews")
    op.drop_index("ix_entity_reviews_created_at", table_name="entity_reviews")
    op.drop_index("ix_entity_reviews_promoted_skill_id", table_name="entity_reviews")
    op.drop_index("ix_entity_reviews_reviewer_id", table_name="entity_reviews")
    op.drop_index("ix_entity_reviews_unknown_entity_id", table_name="entity_reviews")
    op.drop_index("ix_entity_reviews_id", table_name="entity_reviews")
    op.drop_table("entity_reviews")

    op.drop_index("ix_unknown_entities_created_by", table_name="unknown_entities")
    op.drop_index("ix_unknown_entities_updated_at", table_name="unknown_entities")
    op.drop_index("ix_unknown_entities_created_at", table_name="unknown_entities")
    op.drop_index("ix_unknown_entities_canonical_skill_id", table_name="unknown_entities")
    op.drop_index("ix_unknown_entities_candidate_id", table_name="unknown_entities")
    op.drop_index("ix_unknown_entities_status", table_name="unknown_entities")
    op.drop_index("ix_unknown_entities_normalized_value", table_name="unknown_entities")
    op.drop_index("ix_unknown_entities_id", table_name="unknown_entities")
    op.drop_table("unknown_entities")
