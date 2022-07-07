from typing import Optional, List, Any, Dict, TypeVar, Generic, Union

from pydantic import BaseModel
from enum import Enum


class ControlType(str, Enum):
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

    def dict(self, *args, **kwargs) -> "DictStrAny":
        original_dict = super().dict(*args, **kwargs)

        return {"label": original_dict["display_name"], "value": original_dict["value"]}


class BaseControl(BaseModel):
    """
    TODO write documentation
    """

    name: str
    display_name: str
    description: Optional[str]
    type: ControlType

    def dict(self, *args, **kwargs) -> "DictStrAny":
        original_dict = super().dict(*args, **kwargs)

        return {
            "name": original_dict["name"],
            "displayName": original_dict["display_name"],
            "description": original_dict["description"],
            "type": original_dict["type"],
        }


class FloatControl(BaseControl):
    """
    TODO write documentation
    """

    type: ControlType = ControlType.FLOAT
    default: float
    min: Optional[float]
    max: Optional[float]


class OptionsControl(BaseControl, Generic[OptionT]):
    """
    TODO write documentation
    """

    type: ControlType = ControlType.OPTIONS
    is_multi: bool
    default: List[OptionT]
    options: Dict[OptionT, Option[OptionT]]

    def dict(self, *args, **kwargs) -> "DictStrAny":
        original_dict = super().dict(*args, **kwargs)

        return {
            "type": original_dict["type"],
            "isMulti": original_dict["is_multi"],
            "default": [
                original_dict["options"][option] for option in original_dict["default"]
            ],
            "options": list(original_dict["options"].values()),
        }


class StringControl(BaseControl):
    """
    TODO write documentation
    """

    type: ControlType = ControlType.STRING
    default: str


class IntegerControl(BaseControl):
    """
    TODO write documentation
    """

    type: ControlType = ControlType.INT
    default: int
    min: Optional[int]
    max: Optional[int]


class BooleanControl(BaseControl):
    """
    TODO write documentation
    """

    type: ControlType = ControlType.BOOL
    default: bool


class ControlPanel(BaseModel):
    """
    TODO write documentation
    """

    parameters: List[BaseControl]
