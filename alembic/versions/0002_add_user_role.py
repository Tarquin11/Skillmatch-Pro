"""add user role

Revision ID: 0002_add_user_role
Revises: 0001_initial_schema
Create Date: 2026-02-22
"""

from alembic import op
import sqlalchemy as sa

revision = '0002_add_user_role'
down_revision = '0001_initial_schema'
branch_labels = None
depends_on = None

def upgrade() -> None:
    with op.batch_alter_table('users') as batch_op:
        batch_op.add_column(
            sa.Column('role', sa.String(length=20), nullable=False, server_default='user')
        )

def downgrade() -> None:
    with op.batch_alter_table('users') as batch_op:
        batch_op.drop_column('role')