import logging
from app.db.database import Base, engine, ensure_runtime_support_tables, ensure_sqlite_legacy_employee_columns
from app.models.employee import Employee
from app.models.skill import Skill
from app.models.active_learning import EntityReview, UnknownEntity

logger = logging.getLogger(__name__)

def init_db():
    logger.info("Creating database tables.")
    Base.metadata.create_all(bind=engine)
    ensure_sqlite_legacy_employee_columns()
    ensure_runtime_support_tables()
    logger.info("Tables created successfully.")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    init_db()
