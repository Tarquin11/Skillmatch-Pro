"""add departements table

Revision ID: 0003_add_departements_table
Revises: 0002_add_user_role
Create Date: 2026-02-27
"""

from alembic import op
import sqlalchemy as sa


revision = "0003_add_departements_table"
down_revision = "0002_add_user_role"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "departements",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_departements_id", "departements", ["id"], unique=False)
    op.create_index("ix_departements_name", "departements", ["name"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_departements_name", table_name="departements")
    op.drop_index("ix_departements_id", table_name="departements")
    op.drop_table("departements")
