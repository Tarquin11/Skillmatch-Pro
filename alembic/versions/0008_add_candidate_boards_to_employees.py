"""add candidate boards to employees

Revision ID: 0008_add_candidate_boards_to_employees
Revises: 0007_add_phone_to_employees
Create Date: 2026-04-22
"""

from alembic import op
import sqlalchemy as sa


revision = "0008_add_candidate_boards_to_employees"
down_revision = "0007_add_phone_to_employees"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("employees", schema=None) as batch_op:
        batch_op.add_column(sa.Column("candidate_certifications", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("candidate_projects", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("employees", schema=None) as batch_op:
        batch_op.drop_column("candidate_projects")
        batch_op.drop_column("candidate_certifications")
