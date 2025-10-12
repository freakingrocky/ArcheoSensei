from app.main import app
from fastapi.testclient import TestClient


def test_health():
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
