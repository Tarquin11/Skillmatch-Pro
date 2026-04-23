import logging

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
logger = logging.getLogger(__name__)


def ensure_sqlite_legacy_employee_columns() -> None:
    """Backfill columns added after initial local DB creation for SQLite users.

    Some local setups rely on `Base.metadata.create_all()`, which does not alter
    existing tables. This helper keeps older `skillmatch.db` files compatible
    with the current ORM model without requiring a full Alembic pipeline.
    """
    if engine.dialect.name != "sqlite":
        return

    inspector = inspect(engine)
    if "employees" not in inspector.get_table_names():
        return

    existing_columns = {str(col.get("name") or "") for col in inspector.get_columns("employees")}
    statements: list[str] = []
    if "candidate_certifications" not in existing_columns:
        statements.append("ALTER TABLE employees ADD COLUMN candidate_certifications TEXT")
    if "candidate_projects" not in existing_columns:
        statements.append("ALTER TABLE employees ADD COLUMN candidate_projects TEXT")

    if not statements:
        return

    with engine.begin() as connection:
        for statement in statements:
            connection.exec_driver_sql(statement)
    logger.info("Applied SQLite employee schema compatibility patch: %s", ", ".join(statements))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
