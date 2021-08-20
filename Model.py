import abc
from typing import List
from PIL import Image


class BaseModel:

    @abc.abstractmethod
    def fit(self, images: List[Image.Image], labeled_images: List[Image.Image]):
        raise NotImplementedError("The fit method must be implemented")

    @abc.abstractmethod
    def predict(self, image: Image.Image):
        raise NotImplementedError("The predict method must be implemented")
