from typing import TypeVar, Generic, Optional, Any, Union, Tuple

from pydantic.generics import GenericModel

from actableai.parameters.base import BaseParameter
from actableai.parameters.type import ParameterType
from actableai.parameters.validation import OutOfRangeError, ParameterValidationErrors

NumericT = TypeVar("NumericT", float, int)


class NumericParameter(BaseParameter, GenericModel, Generic[NumericT]):
    """Simple Numeric parameter (either integer or float)."""

    default: NumericT
    min: Optional[NumericT]
    max: Optional[NumericT]

    def validate_parameter(self, value: Any) -> ParameterValidationErrors:
        """ "Validate value using the current parameter.

        Args:
            value: Value to validate.

        Returns:
            ParameterValidationErrors object containing the validation errors.
        """
        errors = super().validate_parameter(value)
        if len(errors) > 0:
            return errors

        if (self.min is not None and value < self.min) or (
            self.max is not None and value >= self.max
        ):
            errors.add_error(
                OutOfRangeError(
                    parameter_name=self.name, min=self.min, max=self.max, given=value
                )
            )

        return errors


class NumericRangeSpace(BaseParameter, GenericModel, Generic[NumericT]):
    """Simple Numeric range space (either integer or float)."""

    default: Union[NumericT, Tuple[NumericT, NumericT]]
    min: Optional[NumericT]
    max: Optional[NumericT]

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

        if not isinstance(value, (list, tuple)):
            value = [value]

        for val in value:
            if (self.min is not None and val < self.min) or (
                self.max is not None and val >= self.max
            ):
                errors.add_error(
                    OutOfRangeError(
                        parameter_name=self.name,
                        min=self.min,
                        max=self.max,
                        given=val,
                    )
                )

        return errors

    def process_parameter(self, value: Any) -> Any:
        """Process a value using the current parameter.

        Args:
            value: Value to process.

        Returns:
            Processed value.
        """
        if isinstance(value, list):
            value = tuple(value)

        if isinstance(value, tuple) and len(value) == 1:
            value = value[0]

        return value


class FloatParameter(NumericParameter[float]):
    """Simple Float parameter."""

    parameter_type: ParameterType = ParameterType.FLOAT


class IntegerParameter(NumericParameter[int]):
    """Simple Integer parameter."""

    parameter_type: ParameterType = ParameterType.INT


class FloatRangeSpace(NumericRangeSpace[float]):
    """Simple Float range space parameter."""

    parameter_type: ParameterType = ParameterType.FLOAT_RANGE


class IntegerRangeSpace(NumericRangeSpace[int]):
    """Simple Int range space parameter."""

    parameter_type: ParameterType = ParameterType.INT_RANGE
