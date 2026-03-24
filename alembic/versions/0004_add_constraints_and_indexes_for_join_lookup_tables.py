"""add constraints and indexes for join/lookup-heavy tables

Revision ID: 0004_add_constraints_and_indexes_for_join_lookup_tables
Revises: 0003_add_departements_table
Create Date: 2026-03-24
"""

from alembic import op
import sqlalchemy as sa

revision = "0004_add_constraints_and_indexes_for_join_lookup_tables"
down_revision = "0003_add_departements_table"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("employee_skills", schema=None) as batch_op:
        batch_op.create_unique_constraint(
            "uq_employee_skills_employee_skill",
            ["employee_id", "skill_id"],
        )
        batch_op.create_check_constraint(
            "ck_employee_skills_level_range",
            "level IS NULL OR (level BETWEEN 1 AND 5)",
        )
        batch_op.create_index("ix_employee_skills_employee_id", ["employee_id"], unique=False)
        batch_op.create_index("ix_employee_skills_skill_id", ["skill_id"], unique=False)

    with op.batch_alter_table("job_skills", schema=None) as batch_op:
        batch_op.create_unique_constraint(
            "uq_job_skills_job_skill",
            ["job_id", "skill_id"],
        )
        batch_op.create_check_constraint(
            "ck_job_skills_required_level_range",
            "required_level IS NULL OR (required_level BETWEEN 1 AND 5)",
        )
        batch_op.create_check_constraint(
            "ck_job_skills_weight_positive",
            "weight IS NULL OR weight > 0",
        )
        batch_op.create_index("ix_job_skills_job_id", ["job_id"], unique=False)
        batch_op.create_index("ix_job_skills_skill_id", ["skill_id"], unique=False)

    with op.batch_alter_table("employees", schema=None) as batch_op:
        batch_op.create_index("ix_employees_departement", ["departement"], unique=False)
        batch_op.create_index("ix_employees_position", ["position"], unique=False)

    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.create_index("ix_users_role", ["role"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_index("ix_users_role")

    with op.batch_alter_table("employees", schema=None) as batch_op:
        batch_op.drop_index("ix_employees_position")
        batch_op.drop_index("ix_employees_departement")

    with op.batch_alter_table("job_skills", schema=None) as batch_op:
        batch_op.drop_index("ix_job_skills_skill_id")
        batch_op.drop_index("ix_job_skills_job_id")
        batch_op.drop_constraint("ck_job_skills_weight_positive", type_="check")
        batch_op.drop_constraint("ck_job_skills_required_level_range", type_="check")
        batch_op.drop_constraint("uq_job_skills_job_skill", type_="unique")

    with op.batch_alter_table("employee_skills", schema=None) as batch_op:
        batch_op.drop_index("ix_employee_skills_skill_id")
        batch_op.drop_index("ix_employee_skills_employee_id")
        batch_op.drop_constraint("ck_employee_skills_level_range", type_="check")
        batch_op.drop_constraint("uq_employee_skills_employee_skill", type_="unique")
