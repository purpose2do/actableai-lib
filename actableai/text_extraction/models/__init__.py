from typing import Type, Dict

from actableai.text_extraction.models.base import BaseTextExtractionModel, Model
from .openai import OpenAI

model_dict: Dict[Model, Type[BaseTextExtractionModel]] = {
    Model.openai: OpenAI,
}
