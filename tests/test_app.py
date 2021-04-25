from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.predictor import Predictor


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.mark.parametrize("file", ["3.png", "5.png", "63.png", "65.png", "172.png"])
def test_predict(file: str, data_dir: Path, client: TestClient):
    with (data_dir / file).open("rb") as f:
        response = client.post("/v1/predict", files={"file": f})
    assert response.status_code == 200


def test_get_predictor(client: TestClient):
    response = client.get("/v1/predictor")
    assert response.status_code == 200
    assert response.json()["name"] == Predictor.mnist_model.value
