from typing import (
    TypeVar,
    Generic,
    List,
    Dict,
    Any,
    Union,
    get_type_hints,
)

from pydantic import validator
from pydantic.generics import GenericModel
from pydantic.typing import get_args

from actableai.parameters.base import BaseParameter
from actableai.parameters.type import ParameterType
from actableai.parameters.validation import (
    InvalidKeyError,
    ParameterValidationErrors,
    ParameterValidationError,
    ParameterTypeError,
)

OptionT = TypeVar("OptionT")


class Option(GenericModel, Generic[OptionT]):
    """
    TODO write documentation
    """

    display_name: str
    value: OptionT


class OptionsParameter(BaseParameter, GenericModel, Generic[OptionT]):
    """
    TODO write documentation
    """

    # TODO check if all default are in options

    parameter_type: ParameterType = ParameterType.OPTIONS
    default: Union[str, List[str]]
    # Automatic dynamic field
    dict_parameter: bool = False
    is_multi: bool
    options: Dict[str, Option[OptionT]]

    @validator("dict_parameter", pre=True, always=True)
    def set_dict_parameter(cls, value):
        """
        TODO write documentation
        """
        return hasattr(cls._get_option_type(), "validate_parameter")

    @validator("default", pre=True, always=True)
    def set_default(cls, value):
        """
        TODO write documentation
        """
        if not isinstance(value, list):
            value = [value]

        return value

    @classmethod
    def _get_option_type(cls):
        """
        TODO write documentation
        """
        # Trick to get the generic type, impossible to use traditional method, see:
        # https://github.com/pydantic/pydantic/issues/3559
        return get_type_hints(get_args(get_type_hints(cls)["options"])[1])["value"]

    def get_available_options(self):
        """
        TODO write documentation
        """
        # We use set here for faster search
        return {option.value for option in self.options.values()}

    def validate_parameter(self, value: Any) -> ParameterValidationErrors:
        """
        TODO write documentation
        """
        errors = super().validate_parameter(value)
        if len(errors) > 0:
            return errors

        is_list = isinstance(value, (list, tuple))
        is_dict = isinstance(value, dict)

        if not is_list and not is_dict:
            is_list = True
            value = [value]

        # Validate type of value
        if self.dict_parameter and not is_dict:
            errors.add_error(
                ParameterTypeError(
                    parameter_name=self.name,
                    expected_type="dict",
                    given_type=str(type(value)),
                )
            )
        elif not self.dict_parameter and not is_list:
            errors.add_error(
                ParameterTypeError(
                    parameter_name=self.name,
                    expected_type="list",
                    given_type=str(type(value)),
                )
            )

        # Stop if any errors
        if len(errors) > 0:
            return errors

        if not self.is_multi and len(value) > 1:
            errors.add_error(
                ParameterValidationError(
                    parameter_name=self.name,
                    message=f"Only one element can be selected, list of len {len(value)} has been given.",
                )
            )

            return errors
        elif len(value) <= 0:
            errors.add_error(
                ParameterValidationError(
                    parameter_name=self.name,
                    message="At least one element needs to be selected.",
                )
            )

        # If the parameters need to be validated (call the `validate_parameter`
        #   function)
        if self.dict_parameter:
            for val_name, val in value.items():
                # Check if the option exists
                if val_name not in self.options:
                    errors.add_error(
                        InvalidKeyError(
                            parameter_name=self.name,
                            key=val_name,
                        )
                    )
                    continue

                # Validate the parameter
                option = self.options[val_name].value
                errors.add_errors(option.validate_parameter(val))

        # Just need to check if the parameter is an available option
        else:
            available_options = self.get_available_options()

            for val in value:
                if val not in available_options:
                    errors.add_error(InvalidKeyError(parameter_name=self.name, key=val))

        return errors

    def process_parameter(self, value: Any) -> Any:
        """
        TODO write documentation
        """
        if self.dict_parameter:
            new_value = {}
            for val_name, val in value.items():
                option = self.options[val_name].value
                new_value[val_name] = option.process_parameter(val)

            value = new_value
        else:
            if isinstance(value, list):
                value = tuple(value)

            if isinstance(value, tuple) and (not self.is_multi or len(value) == 1):
                value = value[0]

        return value

    def get_default(self):
        """
        TODO write documentation
        """
        default_list = self.default
        if not isinstance(default_list, (list, tuple)):
            default_list = [default_list]

        if self.dict_parameter:
            default = {option_name: {} for option_name in default_list}
        else:
            default = [self.options[option_name].value for option_name in default_list]

        return self.process_parameter(default)


class OptionsSpace(OptionsParameter[OptionT], Generic[OptionT]):
    """
    TODO write documentation
    """

    is_multi: bool = True
