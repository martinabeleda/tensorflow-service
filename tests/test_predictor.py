from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from app.predictor import PredictionFunctor, Predictor

_PREDICTORS = [
    PredictionFunctor(Predictor.mnist_dropout),
    PredictionFunctor(Predictor.mnist_model),
    PredictionFunctor(Predictor.mnist_cnn),
]


class TestPredictionFunctor:
    @pytest.mark.parametrize("predictor", _PREDICTORS)
    def test_load_predictor(self, predictor: PredictionFunctor):
        assert isinstance(predictor, PredictionFunctor)
        assert issubclass(type(predictor.model), tf.keras.models.Model)

    @pytest.mark.parametrize("predictor", _PREDICTORS)
    @pytest.mark.parametrize("file", ["3.png", "5.png", "63.png", "65.png", "172.png"])
    def test_predict(self, file: str, data_dir: Path, predictor: PredictionFunctor):
        prediction, uncertainty = predictor(data_dir / file)
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == (10,)
        assert uncertainty.shape == (10,)

    @pytest.mark.parametrize("predictor", _PREDICTORS)
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

    @pytest.mark.parametrize("predictor", _PREDICTORS)
    def test_model_is_grayscale(self, predictor: PredictionFunctor):
        channels = {Predictor.mnist_model: True, Predictor.mnist_dropout: True, Predictor.mnist_cnn: False}
        assert predictor._model_is_grayscale() == channels[predictor.predictor]
