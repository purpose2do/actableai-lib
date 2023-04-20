from abc import ABC
from enum import unique, Enum
from typing import Iterable

from PIL.Image import Image

from actableai.models.base import AAIParametersModel


class BaseOCRModel(AAIParametersModel[Iterable[Image], Iterable[str]], ABC):
    has_fit = False
    has_transform = True
    has_predict = False


@unique
class Model(str, Enum):
    """Enum representing the different model available."""

    tesseract = "tesseract"
