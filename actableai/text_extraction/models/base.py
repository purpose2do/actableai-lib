from abc import ABC
from enum import unique, Enum
from typing import Iterable, List

from actableai.models.base import AAIParametersModel
from actableai.parameters.base import BaseParameter
from actableai.parameters.string import StringListParameter
from actableai.utils.typing import JSONType


class BaseTextExtractionModel(
    AAIParametersModel[Iterable[str], Iterable[JSONType]], ABC
):
    has_fit = False
    has_predict = True
    has_transform = False

    @staticmethod
    def get_base_parameters() -> List[BaseParameter]:
        return [
            StringListParameter(
                name="fields_to_extract",
                display_name="Information to Extract",
                description="List of fields to extract",
                min_len=0,
            ),
        ]


@unique
class Model(str, Enum):
    openai = "openai"
