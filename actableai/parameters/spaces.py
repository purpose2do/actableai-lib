from typing import Optional, List, Any, Dict, TypeVar, Generic, Union, Tuple

from pydantic import BaseModel, root_validator
from pydantic.generics import GenericModel

from actableai.parameters.parameters import (
    ParameterType,
    OptionT,
    Option,
    BaseParameter,
    OptionsParameter,
)


class OptionsSpace(OptionsParameter[OptionT], Generic[OptionT]):
    """
    TODO write documentation
    """

    is_multi: bool = True


class FloatRangeSpace(BaseParameter):
    """
    TODO write documentation
    """

    parameter_type: ParameterType = ParameterType.FLOAT_RANGE
    default: Union[float, Tuple[float, float]]
    min: Optional[float]
    max: Optional[float]


class IntegerRangeSpace(BaseParameter):
    """
    TODO write documentation
    """

    parameter_type: ParameterType = ParameterType.INT_RANGE
    default: Union[int, Tuple[int, int]]
    min: Optional[int]
    max: Optional[int]


class SearchSpace(BaseModel):
    """
    TODO write documentation
    """

    space: List[BaseParameter]
