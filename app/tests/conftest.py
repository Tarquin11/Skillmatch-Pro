import os
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.main import app as fastapi_app  
from app.db.database import Base, get_db


os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_bootstrap.db")

@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    fastapi_app.dependency_overrides[get_db] = override_get_db

    with TestClient(fastapi_app) as c:
        yield c

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