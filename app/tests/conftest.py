import os
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 1. Setup environment variables before other imports
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_bootstrap.db")

# 2. Corrected imports
from app.main import app as fastapi_app  # Imported as fastapi_app
from app.db.database import Base, get_db

# Ensure all models are registered in Base metadata
import app.models.user  
import app.models.employee  
import app.models.job  
import app.models.skill  
import app.models.Employee_skill  
import app.models.departement  

@pytest.fixture()
def client(tmp_path):
    # Create a unique database for each test function
    db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Create tables
    Base.metadata.create_all(bind=engine)

    # Database dependency override
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    # FIX: Use 'fastapi_app' instead of 'app'
    fastapi_app.dependency_overrides[get_db] = override_get_db

    # FIX: Use 'fastapi_app' in the TestClient
    with TestClient(fastapi_app) as c:
        yield c

    # Cleanup after test completes
    fastapi_app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()

def _signup_and_login(client: TestClient, role: str):
    email = f"{role}_{uuid.uuid4().hex[:8]}@example.com"
    password = "Test123!"

    # Create user
    r = client.post(
        "/auth/signup",
        json={"email": email, "password": password, "role": role},
    )
    assert r.status_code == 200, r.text

    # Log in
    r = client.post(
        "/auth/login",
        data={"username": email, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]

    return {"Authorization": f"Bearer {token}"}

@pytest.fixture()
def admin_auth(client):
    return _signup_and_login(client, "admin")

@pytest.fixture()
def user_auth(client):
    return _signup_and_login(client, "user")