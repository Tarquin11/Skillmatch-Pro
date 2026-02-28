import logging
from app.db.database import engine, Base
from app.models.employee import Employee
from app.models.skill import Skill

logger = logging.getLogger(__name__)

def init_db():
    logger.info("Creating database tables.")
    Base.metadata.create_all(bind=engine)
    logger.info("Tables created successfully.")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    init_db()
