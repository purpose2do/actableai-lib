from enum import Enum
from typing import TypeVar, Generic, Dict, Any, Optional, List

from pydantic import BaseModel, root_validator
from pydantic.generics import GenericModel


class ParameterType(str, Enum):
    """
    TODO write documentation
    """

    BOOL = "bool"
    INT = "int"
    INT_RANGE = "int_range"
    FLOAT = "float"
    FLOAT_RANGE = "float_range"
    OPTIONS = "options"


OptionT = TypeVar("OptionT")
NumberT = TypeVar("NumberT", float, int)


class BaseParameter(BaseModel):
    """
    TODO write documentation
    """

    name: str
    display_name: str
    description: Optional[str]
    parameter_type: ParameterType


class Option(BaseModel, Generic[OptionT]):
    """
    TODO write documentation
    """

    display_name: str
    value: OptionT

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        output_dict = super().dict(*args, **kwargs).copy()

        output_dict["_label"] = output_dict["display_name"]

        return output_dict


class OptionsParameter(BaseParameter, GenericModel, Generic[OptionT]):
    """
    TODO write documentation
    """

    parameter_type: ParameterType = ParameterType.OPTIONS
    is_multi: bool
    default: List[OptionT]
    options: Dict[OptionT, Option[OptionT]]

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        output_dict = super().dict(*args, **kwargs).copy()

        output_dict["_default"] = [
            output_dict["options"][option] for option in output_dict["default"]
        ]
        output_dict["_options"] = list(output_dict["options"].values())

        return output_dict


class NumberParameter(BaseParameter, GenericModel, Generic[NumberT]):
    """
    TODO write documentation
    """

    parameter_type: ParameterType = ParameterType.FLOAT
    default: NumberT
    min: Optional[NumberT]
    max: Optional[NumberT]

    @root_validator(skip_on_failure=True)
    def set_type(cls, values):
        type_val = values["default"]

        if isinstance(type_val, int):
            values["parameter_type"] = ParameterType.INT
        elif isinstance(type_val, float):
            values["parameter_type"] = ParameterType.FLOAT
        else:
            raise TypeError(
                "Wrong type for 'parameter_type', must be a float or an int"
            )

        return values


class BooleanParameter(BaseParameter):
    """
    TODO write documentation
    """

    parameter_type: ParameterType = ParameterType.BOOL
    default: bool


class Parameters(BaseModel):
    """
    TODO write documentation
    """

    parameters: List[BaseParameter]
