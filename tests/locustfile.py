from pathlib import Path

from locust import HttpUser, TaskSet, between, task

from app.predictor import Predictor


class MNISTPredict(TaskSet):
    @task
    def predict(self):
        test_image = Path(__file__).parent.absolute() / "data/3.png"
        with test_image.open("rb") as f:
            response = self.client.post("/v1/predict", files={"file": f})
        assert response.status_code == 200
        data = response.json()
        assert len(data["prediction"]) == 10
        assert len(data["uncertainty"]) == 10
        assert data["predictorName"] == Predictor.mnist_dropout.value


class MNISTLoadTest(HttpUser):
    tasks = [MNISTPredict]
    host = "http://127.0.0.1"
    wait_time = between(1, 5)
