from typing import Optional, List, Any, Dict, TypeVar, Generic, Union, Tuple

from pydantic import BaseModel, root_validator
from pydantic.generics import GenericModel

from actableai.parameters.parameters import (
    ParameterType,
    OptionT,
    Option,
    BaseParameter,
    NumberT,
    OptionsParameter,
)


class OptionsSpace(OptionsParameter[OptionT], Generic[OptionT]):
    """
    TODO write documentation
    """

    is_multi: bool = True


class RangeSpace(BaseParameter, GenericModel, Generic[NumberT]):
    """
    TODO write documentation
    """

    parameter_type: ParameterType = ParameterType.FLOAT_RANGE
    default: Union[NumberT, Tuple[NumberT, NumberT]]
    min: Optional[NumberT]
    max: Optional[NumberT]

    @root_validator(skip_on_failure=True)
    def set_type(cls, values):
        type_val = values["default"]
        if isinstance(type_val, tuple):
            type_val = type_val[0]

        if isinstance(type_val, int):
            values["parameter_type"] = ParameterType.INT_RANGE
        elif isinstance(type_val, float):
            values["parameter_type"] = ParameterType.FLOAT_RANGE
        else:
            raise TypeError(
                "Wrong type for 'parameter_type', must be a float or an int"
            )

        return values


class SearchSpace(BaseModel):
    """
    TODO write documentation
    """

    space: List[BaseParameter]
