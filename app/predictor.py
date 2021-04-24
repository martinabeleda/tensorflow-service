from __future__ import annotations

import dataclasses
from enum import Enum
from pathlib import Path
from typing import IO, Tuple, Union

from PIL import Image
import numpy as np
import tensorflow as tf


class Predictor(str, Enum):
    """The predictors that we currently support"""

    mnist_model = "mnist_model"
    flowers_colour = "flowers_colour"


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

    @property
    def image_shape(self) -> Union[Tuple[int, int], Tuple[int, int, int]]:
        return self.model.input_shape[1:]

    @property
    def image_context(self) -> Tuple[int, int]:
        return self.image_shape if len(self.image_shape) == 2 else self.image_shape[:2]

    def _load_image_as_array(self, file: IO) -> np.ndarray:
        """Loads an image from file and applies pre-processing steps to conform to the model input

        Args:
            file (IO): A file-like object pointing to the image file

        Returns:
            np.ndarray: The image as an array that is compatible with the model
        """
        with Image.open(file) as image:
            if self._model_is_grayscale():
                image = image.convert("L")
            else:
                image = image.convert("RGB")
            if not image.size == self.image_context:
                image = image.resize(self.image_context)
        return np.asarray(image)

    def _model_is_grayscale(self) -> bool:
        return len(self.image_shape) == 2

    @staticmethod
    def _load_predictor(predictor: Predictor) -> tf.keras.Model:
        model_path = Path(__file__).parent.resolve() / "predictors" / predictor.value
        return tf.keras.models.load_model(str(model_path))
