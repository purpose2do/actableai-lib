from abc import ABC, abstractmethod
from typing import Optional, Any, Tuple

from pydantic import BaseModel

from actableai.parameters.type import ParameterType
from actableai.parameters.validation import (
    ParameterTypeError,
    ParameterValidationErrors,
    ParameterValidationError,
)


class NamedParameter(BaseModel):
    """Base class representing a parameter with a name."""

    name: str
    display_name: str
    description: Optional[str]


class ValidatableParameter(ABC):
    """Abstract class representing a validatable parameter."""

    @abstractmethod
    def validate_parameter(self, value: Any) -> ParameterValidationErrors:
        """Validate value using the current parameter.

        Args:
            value: Value to validate.

        Returns:
            ParameterValidationErrors object containing the validation errors.
        """
        raise NotImplementedError()


class ProcessableParameter(ValidatableParameter, ABC):
    """Abstract class representing a processable parameter (also validatable)."""

    @abstractmethod
    def process_parameter(self, value: Any) -> Any:
        """Process a value using the current parameter.

        Args:
            value: Value to process.

        Returns:
            Processed value.
        """
        raise NotImplementedError()

    def validate_process_parameter(
        self, value: Any
    ) -> Tuple[ParameterValidationErrors, Any]:
        """Validate and process value using current parameter.

        Args:
            value: Value to validate and process.

        Returns:
            - ParameterValidationErrors object containing the validation errors.
            - Processed value.
        """
        validation_errors = self.validate_parameter(value)

        if len(validation_errors) > 0:
            return validation_errors, None

        return validation_errors, self.process_parameter(value)


class BaseParameter(NamedParameter, ProcessableParameter):
    """Base class representing a parameter."""

    parameter_type: ParameterType
    default: Any
    hidden: bool = False

    def validate_parameter(self, value: Any) -> ParameterValidationErrors:
        """Validate value using the current parameter.

        Args:
            value: Value to validate.

        Returns:
            ParameterValidationErrors object containing the validation errors.
        """
        errors = ParameterValidationErrors(parameter_name=self.name)

        type_valid = True

        # Check type
        if self.parameter_type == ParameterType.BOOL:
            type_valid = isinstance(value, bool)
        elif self.parameter_type == ParameterType.INT:
            type_valid = isinstance(value, int)
        elif self.parameter_type == ParameterType.FLOAT:
            type_valid = isinstance(value, (int, float))
        elif self.parameter_type == ParameterType.INT_RANGE:
            if not isinstance(value, (list, tuple)):
                value = [value]

            for val in value:
                if not isinstance(val, int):
                    type_valid = False
                    break
        elif self.parameter_type == ParameterType.FLOAT_RANGE:
            if not isinstance(value, (list, tuple)):
                value = [value]

            for val in value:
                if not isinstance(val, (int, float)):
                    type_valid = False
                    break

        if not type_valid:
            errors.add_error(
                ParameterTypeError(
                    parameter_name=self.name,
                    expected_type=self.parameter_type,
                    given_type=str(type(value)),
                )
            )

        # Check range len
        if (
            (
                self.parameter_type == ParameterType.INT_RANGE
                or self.parameter_type == ParameterType.FLOAT_RANGE
            )
            and type_valid
            and (len(value) <= 0 or len(value) > 2)
        ):
            errors.add_error(
                ParameterValidationError(
                    parameter_name=self.name,
                    message=f"A list of one or two elements is expected, list of len {len(value)} has been given.",
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
        return value

    def get_default(self) -> Any:
        """Return default value for the parameter.

        Returns:
            Default value.
        """
        return self.default
