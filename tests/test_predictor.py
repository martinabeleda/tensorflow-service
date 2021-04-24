from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from app.predictor import PredictionFunctor, Predictor


@pytest.fixture
def predictor() -> PredictionFunctor:
    return PredictionFunctor(Predictor.mnist_model)


class TestPredictionFunctor:
    def test_load_predictor(self, predictor: PredictionFunctor):
        assert isinstance(predictor, PredictionFunctor)
        assert issubclass(type(predictor.model), tf.keras.models.Model)
        assert predictor.predictor == Predictor.mnist_model

    @pytest.mark.parametrize("file", ["3.png", "5.png", "63.png", "65.png", "172.png"])
    def test_predict(self, file: str, data_dir: Path, predictor: PredictionFunctor):
        prediction = predictor(data_dir / file)
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == (1, 10)

    @pytest.mark.parametrize(
        "file",
        [
            "colour_320_232.jpg",
            "colour_320_240.png",
            "colour_500_332.jpg",
            "grayscale_999_893.png",
        ],
    )
    def test_load_image_as_array(self, file: str, data_dir: Path, predictor: PredictionFunctor):
        image = predictor._load_image_as_array(data_dir / file)
        assert image.shape == predictor.image_shape

    def test_model_is_grayscale(self, predictor: PredictionFunctor):
        assert predictor._model_is_grayscale()
