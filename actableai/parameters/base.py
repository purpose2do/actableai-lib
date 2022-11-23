from abc import ABC, abstractmethod
from typing import Optional, Any, Tuple

from pydantic import BaseModel

from actableai.parameters.type import ParameterType
from actableai.parameters.validation import (
    ParameterValidationErrors,
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
    default: Optional[Any] = None
    hidden: bool = False

    def validate_parameter(self, value: Any) -> ParameterValidationErrors:
        """Validate value using the current parameter.

        Args:
            value: Value to validate.

        Returns:
            ParameterValidationErrors object containing the validation errors.
        """
        return ParameterValidationErrors(parameter_name=self.name)

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
        return self.process_parameter(self.default)
