from typing import Any, Generic, TypeVar, get_type_hints, Type, Optional

from pydantic import validator
from pydantic.generics import GenericModel
from pydantic.typing import get_args

from actableai.parameters.base import BaseParameter
from actableai.parameters.type import ValueType, ParameterType
from actableai.parameters.validation import (
    ParameterValidationErrors,
    ParameterTypeError,
)

ValueT = TypeVar("ValueT", bool, int, float, str)


class ValueParameter(BaseParameter, GenericModel, Generic[ValueT]):
    """
    TODO write documentation
    """

    parameter_type: ParameterType = ParameterType.VALUE
    default: Optional[ValueT] = None
    # Automatic dynamic field
    value_type: ValueType = None

    @validator("value_type", always=True)
    def set_value_type(cls, value_type: ValueType) -> ValueType:
        """
        TODO write documentation
        """
        val_type = cls._get_value_type()
        if val_type == bool:
            return ValueType.BOOL
        if val_type == int:
            return ValueType.INT
        if val_type == float:
            return ValueType.FLOAT
        if val_type == str:
            return ValueType.STR

        raise ValueError("Invalid generic type.")

    @classmethod
    def _get_value_type(cls) -> Type:
        """Returns the type of the `ValueT`.

        Returns:
            `ValueT` type.
        """
        # Trick to get the generic type, impossible to use traditional method, see:
        # https://github.com/pydantic/pydantic/issues/3559
        return get_args(get_type_hints(cls)["default"])[0]

    def validate_parameter(self, value: Any) -> ParameterValidationErrors:
        """Validate value using the current parameter.

        Args:
            value: Value to validate.

        Returns:
            ParameterValidationErrors object containing the validation errors.
        """
        errors = super().validate_parameter(value)
        if len(errors) > 0:
            return errors

        type_valid = True

        if self.value_type == ValueType.BOOL:
            type_valid = isinstance(value, bool)
        elif self.value_type == ValueType.INT:
            type_valid = isinstance(value, int)
        elif self.value_type == ValueType.FLOAT:
            type_valid = isinstance(value, (int, float))
        elif self.value_type == ValueType.STR:
            type_valid = isinstance(value, str)

        if not type_valid:
            errors.add_error(
                ParameterTypeError(
                    parameter_name=self.name,
                    expected_type=self.value_type,
                    given_type=str(type(value)),
                )
            )

        return errors
