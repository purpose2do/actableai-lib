from typing import Dict, List, Any, Union

from pydantic import validator

from actableai.parameters.base import (
    BaseParameter,
    ProcessableParameter,
    NamedParameter,
)
from actableai.parameters.validation import (
    InvalidKeyError,
    ParameterValidationErrors, ParameterTypeError,
)


class Parameters(NamedParameter, ProcessableParameter):
    """
    TODO write documentation
    """

    parameters: Union[Dict[str, BaseParameter], List[BaseParameter]]

    @validator("parameters", pre=True, always=True)
    def set_parameters(cls, value):
        """
        TODO write documentation
        """
        if isinstance(value, list):
            value = {parameter.name: parameter for parameter in value}

        return value

    def validate_parameter(self, value: Any) -> ParameterValidationErrors:
        """
        TODO write documentation
        """
        errors = ParameterValidationErrors(parameter_name=self.name)

        if not isinstance(value, dict):
            errors.add_error(
                ParameterTypeError(
                    parameter_name=self.name,
                    expected_type="dict",
                    given_type=str(type(value)),
                )
            )

            return errors

        for val_name, val in value.items():
            if val_name not in self.parameters:
                errors.add_error(
                    InvalidKeyError(parameter_name=self.name, key=val_name)
                )
                continue

            errors.add_errors(self.parameters[val_name].validate_parameter(val))

        return errors

    def process_parameter(self, value: Any) -> Any:
        """
        TODO write documentation
        """
        final_parameters = {}

        for parameter_name, parameter in self.parameters.items():
            if parameter_name in value:
                final_parameters[parameter_name] = parameter.process_parameter(
                    value[parameter_name]
                )
            else:
                final_parameters[parameter_name] = self.parameters[
                    parameter_name
                ].get_default()

        return final_parameters
