import pytest
from fastapi.testclient import TestClient
from app.config import settings
from app.main import app, seed_default_tenants_and_docs
from app.db.session import engine, Base


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Ensures test database tables and initial test tenants exist for test duration."""
    settings.SEED_DEMO_DATA = True
    Base.metadata.create_all(bind=engine)
    seed_default_tenants_and_docs(force=True)
    yield


@pytest.fixture(scope="module")
def client():
    """TestClient fixture with active test database."""
    Base.metadata.create_all(bind=engine)
    seed_default_tenants_and_docs(force=True)
    with TestClient(app) as c:
        yield c
