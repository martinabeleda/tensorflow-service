from __future__ import annotations

import dataclasses
from enum import Enum
from pathlib import Path
from typing import IO

from PIL import Image
import numpy as np
import tensorflow as tf


class Predictor(str, Enum):
    """The predictors that we currently support"""

    mnist_model = "mnist_model"


@dataclasses.dataclass(frozen=True)
class PredictionFunctor:

    predictor: Predictor
    model: tf.keras.Model = dataclasses.field(init=False, default=False)

    def __post_init__(self):
        object.__setattr__(self, "model", self._load_predictor(self.predictor))

    def __call__(self, file: IO) -> np.ndarray:
        image = self._load_image_as_array(file)
        image = np.expand_dims(image, axis=0)
        return self.model.predict(image)

    @staticmethod
    def _load_image_as_array(file: IO) -> np.ndarray:
        with Image.open(file) as image:
            return np.asarray(image)

    @staticmethod
    def _load_predictor(predictor: Predictor) -> tf.keras.Model:
        model_path = Path(__file__).parent.resolve() / "predictors" / predictor.value
        return tf.keras.models.load_model(str(model_path))
