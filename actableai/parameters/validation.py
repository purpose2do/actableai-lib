from typing import Optional, Union, List

from pydantic import BaseModel, root_validator

from actableai.data_validation.base import CheckResult, CheckLevels
from actableai.parameters.type import ParameterType


class ParameterValidationError(BaseModel):
    """
    TODO write documentation
    """

    parameter_name: str
    message: Optional[str]
    parent_parameter_list: List[str] = []

    @property
    def path(self):
        """
        TODO write documentation
        """
        path = ""
        for parent in reversed(self.parent_parameter_list):
            path = f"{path}{parent}."

        path = f"{path}{self.parameter_name}"

        return path

    def __str__(self) -> str:
        """
        TODO write documentation
        """
        error = self.path
        if self.message is not None:
            error = f"{error}: {self.message}"
        return error

    def add_parent(self, parameter_name):
        """
        TODO write documentation
        """
        self.parent_parameter_list.append(parameter_name)

    @staticmethod
    def _process_message(message, values):
        """
        TODO write documentation
        """
        if values.get("message") is not None:
            message = f"{message} (values['message'])"

        values["message"] = f"{message}."
        return values


class ParameterValidationErrors(BaseModel):
    """
    TODO write documentation
    """

    parameter_name: str
    validation_error_list: List[ParameterValidationError] = []

    def __len__(self):
        return len(self.validation_error_list)

    def add_error(self, error: ParameterValidationError):
        """
        TODO write documentation
        """
        self.validation_error_list.append(error)

    def add_errors(self, errors: "ParameterValidationErrors"):
        """
        TODO write documentation
        """
        for error in errors.validation_error_list:
            error.add_parent(self.parameter_name)
            self.validation_error_list.append(error)

    def to_check_results(self, name) -> List[CheckResult]:
        """
        TODO write documentation
        """
        check_result_list = []
        for error in self.validation_error_list:
            check_result_list.append(
                CheckResult(name=name, message=str(error), level=CheckLevels.CRITICAL)
            )

        return check_result_list


class ParameterTypeError(ParameterValidationError):
    """
    TODO write documentation
    """

    expected_type: Union[ParameterType, str]
    given_type: str

    @root_validator(pre=True)
    def set_message(cls, values):
        """
        TODO write documentation
        """
        message = f"Incorrect parameter type, given: `{values['given_type']}`, expected: `{values['expected_type']}`"
        return cls._process_message(message, values)


class OutOfRangeError(ParameterValidationError):
    """
    TODO write documentation
    """

    min: Optional[Union[int, float]]
    max: Optional[Union[int, float]]
    given: Union[int, float]

    @root_validator(pre=True)
    def set_message(cls, values):
        """
        TODO write documentation
        """
        message = f"Out of range value, given: `{values['given']}`"

        if values.get("min") is not None:
            message = f"{message}, minimum: `{values['min']}`"
        if values.get("max") is not None:
            message = f"{message}, maximum (excluded): `{values['max']}`"

        return cls._process_message(message, values)


class InvalidKeyError(ParameterValidationError):
    """
    TODO write documentation
    """

    key: str

    @root_validator(pre=True)
    def set_message(cls, values):
        """
        TODO write documentation
        """
        message = f"The key `{values['key']}` is invalid"

        return cls._process_message(message, values)
