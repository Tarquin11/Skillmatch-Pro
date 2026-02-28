from sqlalchemy import Column , Integer, String
from app.db.database import Base

class Departement(Base):
    __tablename__ = "departements"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
      