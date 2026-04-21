"""add phone to employees

Revision ID: 0007_add_phone_to_employees
Revises: 0006_add_auth_hardening_fields_and_refresh_tokens
Create Date: 2026-04-21
"""

from alembic import op
import sqlalchemy as sa


revision = "0007_add_phone_to_employees"
down_revision = "0006_add_auth_hardening_fields_and_refresh_tokens"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("employees", schema=None) as batch_op:
        batch_op.add_column(sa.Column("phone", sa.String(), nullable=True))
        batch_op.create_index("ix_employees_phone", ["phone"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("employees", schema=None) as batch_op:
        batch_op.drop_index("ix_employees_phone")
        batch_op.drop_column("phone")
