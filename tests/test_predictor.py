from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from app.predictor import PredictionFunctor, Predictor


@pytest.fixture
def test_predictor() -> Predictor:
    return Predictor.mnist_model


class TestPredictionFunctor:
    def test_load_predictor(self, test_predictor: Predictor):
        predictor = PredictionFunctor(test_predictor)
        assert isinstance(predictor, PredictionFunctor)
        assert issubclass(type(predictor.model), tf.keras.models.Model)
        assert predictor.predictor == Predictor.mnist_model

    @pytest.mark.parametrize("file", ["3.png", "5.png", "63.png", "65.png", "172.png"])
    def test_predict(self, file: str, data_dir: Path, test_predictor: Predictor):
        filepath = data_dir / file
        predictor = PredictionFunctor(test_predictor)
        prediction = predictor(filepath)
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == (1, 10)
