from typing import Optional, List, Any, Dict, TypeVar, Generic, Union

from pydantic import BaseModel
from enum import Enum


class SpaceType(str, Enum):
    """
    TODO write documentation
    """

    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    OPTIONS = "options"
    STRING = "string"


OptionT = TypeVar("OptionT")


class Option(BaseModel, Generic[OptionT]):
    """
    TODO write documentation
    """

    display_name: str
    value: OptionT

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        original_dict = super().dict(*args, **kwargs)

        return {"label": original_dict["display_name"], "value": original_dict["value"]}


class BaseSpace(BaseModel):
    """
    TODO write documentation
    """

    name: str
    display_name: str
    description: Optional[str]
    type: SpaceType

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        original_dict = super().dict(*args, **kwargs)

        return {
            "name": original_dict["name"],
            "displayName": original_dict["display_name"],
            "description": original_dict["description"],
            "type": original_dict["type"],
        }


class FloatSpace(BaseSpace):
    """
    TODO write documentation
    """

    type: SpaceType = SpaceType.FLOAT
    default: float
    min: Optional[float]
    max: Optional[float]


class OptionsSpace(BaseSpace, Generic[OptionT]):
    """
    TODO write documentation
    """

    type: SpaceType = SpaceType.OPTIONS
    is_multi: bool
    default: List[OptionT]
    options: Dict[OptionT, Option[OptionT]]

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        original_dict = super().dict(*args, **kwargs)

        return {
            "type": original_dict["type"],
            "isMulti": original_dict["is_multi"],
            "default": [
                original_dict["options"][option] for option in original_dict["default"]
            ],
            "options": list(original_dict["options"].values()),
        }


class StringSpace(BaseSpace):
    """
    TODO write documentation
    """

    type: SpaceType = SpaceType.STRING
    default: str


class IntegerSpace(BaseSpace):
    """
    TODO write documentation
    """

    type: SpaceType = SpaceType.INT
    default: int
    min: Optional[int]
    max: Optional[int]


class BooleanSpace(BaseSpace):
    """
    TODO write documentation
    """

    type: SpaceType = SpaceType.BOOL
    default: bool


class SearchSpace(BaseModel):
    """
    TODO write documentation
    """

    parameters: List[BaseSpace]
