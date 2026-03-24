"""add auth hardening fields and refresh tokens

Revision ID: 0006_add_auth_hardening_fields_and_refresh_tokens
Revises: 0005_add_audit_fields_to_core_tables
Create Date: 2026-03-24
"""

from alembic import op
import sqlalchemy as sa

revision = "0006_add_auth_hardening_fields_and_refresh_tokens"
down_revision = "0005_add_audit_fields_to_core_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.add_column(sa.Column("failed_login_attempts", sa.Integer(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("last_failed_login_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("token_version", sa.Integer(), nullable=False, server_default="0"))
        batch_op.create_index("ix_users_locked_until", ["locked_until"], unique=False)
        batch_op.create_index("ix_users_token_version", ["token_version"], unique=False)

    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("token_hash", sa.String(length=128), nullable=False),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("replaced_by_token_hash", sa.String(length=128), nullable=True),
        sa.Column("ip_address", sa.String(length=64), nullable=True),
        sa.Column("user_agent", sa.String(length=512), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash", name="uq_refresh_tokens_token_hash"),
    )
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"], unique=False)
    op.create_index("ix_refresh_tokens_expires_at", "refresh_tokens", ["expires_at"], unique=False)
    op.create_index("ix_refresh_tokens_revoked_at", "refresh_tokens", ["revoked_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_refresh_tokens_revoked_at", table_name="refresh_tokens")
    op.drop_index("ix_refresh_tokens_expires_at", table_name="refresh_tokens")
    op.drop_index("ix_refresh_tokens_user_id", table_name="refresh_tokens")
    op.drop_table("refresh_tokens")

    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_index("ix_users_token_version")
        batch_op.drop_index("ix_users_locked_until")
        batch_op.drop_column("token_version")
        batch_op.drop_column("locked_until")
        batch_op.drop_column("last_failed_login_at")
        batch_op.drop_column("failed_login_attempts")
