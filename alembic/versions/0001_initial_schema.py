"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-02-21 14:37:43
"""

from alembic import op
import sqlalchemy as sa

revision = '0001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        "employees",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("Employee_number", sa.String(), nullable=False),
        sa.Column("first_name", sa.String(), nullable=False),
        sa.Column("last_name", sa.String(), nullable=False),
        sa.Column("full_name", sa.String(), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("gender", sa.String(), nullable=True),
        sa.Column("dob", sa.Date(), nullable=True),
        sa.Column("marital_status", sa.String(), nullable=True),
        sa.Column("citizenship", sa.String(), nullable=True),
        sa.Column("Governorate", sa.String(), nullable=True),
        sa.Column("city", sa.String(), nullable=True),
        sa.Column("zip_code", sa.String(), nullable=True),
        sa.Column("departement", sa.String(), nullable=True),
        sa.Column("position", sa.String(), nullable=True),
        sa.Column("manager_name", sa.String(), nullable=True),
        sa.Column("Employment_status", sa.String(), nullable=True),
        sa.Column("salary", sa.Float(), nullable=True),
        sa.Column("pay_rate", sa.Float(), nullable=True),
        sa.Column("performance_score", sa.String(), nullable=True),
        sa.Column("engagement_survey", sa.Float(), nullable=True),
        sa.Column("emp_satisfaction", sa.Integer(), nullable=True),
        sa.Column("hire_date", sa.Date(), nullable=True),
        sa.Column("termination_date", sa.Date(), nullable=True),
        sa.Column("termination_reason", sa.String(), nullable=True),
        sa.Column("recruitement_source", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_employees_id", "employees", ["id"], unique=False)
    op.create_index("ix_employees_Employee_number", "employees", ["Employee_number"], unique=True)
    op.create_index("ix_employees_email", "employees", ["email"], unique=True)

    op.create_table(
        "skills",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_skills_id", "skills", ["id"], unique=False)
    op.create_index("ix_skills_name", "skills", ["name"], unique=True)

    op.create_table(
        "job_posts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("departement", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_job_posts_id", "job_posts", ["id"], unique=False)
    op.create_index("ix_job_posts_title", "job_posts", ["title"], unique=False)
    op.create_index("ix_job_posts_departement", "job_posts", ["departement"], unique=False)

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_id", "users", ["id"], unique=False)
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "employee_skills",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("employee_id", sa.Integer(), nullable=False),
        sa.Column("skill_id", sa.Integer(), nullable=False),
        sa.Column("level", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["employee_id"], ["employees.id"]),
        sa.ForeignKeyConstraint(["skill_id"], ["skills.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_employee_skills_id", "employee_skills", ["id"], unique=False)

    op.create_table(
        "job_skills",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("skill_id", sa.Integer(), nullable=False),
        sa.Column("required_level", sa.Integer(), nullable=True),
        sa.Column("weight", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["job_id"], ["job_posts.id"]),
        sa.ForeignKeyConstraint(["skill_id"], ["skills.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_job_skills_id", "job_skills", ["id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_job_skills_id", table_name="job_skills")
    op.drop_table("job_skills")

    op.drop_index("ix_employee_skills_id", table_name="employee_skills")
    op.drop_table("employee_skills")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_id", table_name="users")
    op.drop_table("users")

    op.drop_index("ix_job_posts_departement", table_name="job_posts")
    op.drop_index("ix_job_posts_title", table_name="job_posts")
    op.drop_index("ix_job_posts_id", table_name="job_posts")
    op.drop_table("job_posts")

    op.drop_index("ix_skills_name", table_name="skills")
    op.drop_index("ix_skills_id", table_name="skills")
    op.drop_table("skills")

    op.drop_index("ix_employees_email", table_name="employees")
    op.drop_index("ix_employees_Employee_number", table_name="employees")
    op.drop_index("ix_employees_id", table_name="employees")
    op.drop_table("employees")