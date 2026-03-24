"""add audit fields to core tables

Revision ID: 0005_add_audit_fields_to_core_tables
Revises: 0004_add_constraints_and_indexes_for_join_lookup_tables
Create Date: 2026-03-24
"""

from alembic import op
import sqlalchemy as sa


revision = "0005_add_audit_fields_to_core_tables"
down_revision = "0004_add_constraints_and_indexes_for_join_lookup_tables"
branch_labels = None
depends_on = None


AUDITED_TABLES = [
    "employees",
    "skills",
    "job_posts",
    "employee_skills",
    "job_skills",
    "departements",
    "users",
]


def upgrade() -> None:
    for table_name in AUDITED_TABLES:
        with op.batch_alter_table(table_name, schema=None) as batch_op:
            batch_op.add_column(
                sa.Column(
                    "created_at",
                    sa.DateTime(timezone=True),
                    nullable=False,
                    server_default=sa.text("CURRENT_TIMESTAMP"),
                )
            )
            batch_op.add_column(
                sa.Column(
                    "updated_at",
                    sa.DateTime(timezone=True),
                    nullable=False,
                    server_default=sa.text("CURRENT_TIMESTAMP"),
                )
            )
            batch_op.add_column(sa.Column("created_by", sa.Integer(), nullable=True))
            batch_op.create_index(f"ix_{table_name}_created_at", ["created_at"], unique=False)
            batch_op.create_index(f"ix_{table_name}_updated_at", ["updated_at"], unique=False)
            batch_op.create_index(f"ix_{table_name}_created_by", ["created_by"], unique=False)
            batch_op.create_foreign_key(
                f"fk_{table_name}_created_by_users",
                "users",
                ["created_by"],
                ["id"],
            )


def downgrade() -> None:
    for table_name in reversed(AUDITED_TABLES):
        with op.batch_alter_table(table_name, schema=None) as batch_op:
            batch_op.drop_constraint(f"fk_{table_name}_created_by_users", type_="foreignkey")
            batch_op.drop_index(f"ix_{table_name}_created_by")
            batch_op.drop_index(f"ix_{table_name}_updated_at")
            batch_op.drop_index(f"ix_{table_name}_created_at")
            batch_op.drop_column("created_by")
            batch_op.drop_column("updated_at")
            batch_op.drop_column("created_at")

