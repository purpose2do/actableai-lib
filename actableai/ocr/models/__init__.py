from typing import Type, Dict

from actableai.ocr.models.base import BaseOCRModel, Model
from .tesseract import Tesseract

model_dict: Dict[Model, Type[BaseOCRModel]] = {
    Model.tesseract: Tesseract,
}
