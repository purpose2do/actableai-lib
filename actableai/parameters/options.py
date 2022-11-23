from typing import (
    TypeVar,
    Generic,
    List,
    Dict,
    Any,
    Union,
    get_type_hints,
    Set,
    Type,
)

from pydantic import validator, root_validator
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
    """Class to represent a simple option, used in `OptionsParameter`."""

    display_name: str
    value: OptionT


class OptionsParameter(BaseParameter, GenericModel, Generic[OptionT]):
    """Parameter representing options of type `OptionT`"""

    parameter_type: ParameterType = ParameterType.OPTIONS
    default: Union[str, List[str]]
    # Automatic dynamic field
    dict_parameter: bool = False
    is_multi: bool
    options: Dict[str, Option[OptionT]]

    @validator("dict_parameter", always=True)
    def set_dict_parameter(cls, value: bool) -> bool:
        """Set `dict_parameter` value.

        Args:
            value: Current value of `dict_parameter`.

        Returns:
            New `dict_parameter` value.
        """
        return hasattr(cls._get_option_type(), "validate_parameter")

    @validator("default", always=True)
    def set_default(cls, value: Union[str, List[str]]) -> List[str]:
        """Set `default` value.

        Args:
            value: Current value of `default`.

        Returns:
            New `default` value.
        """
        if not isinstance(value, list):
            value = [value]

        return value

    @root_validator(skip_on_failure=True)
    def check_default(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Check default value using the current parameter.

        Args:
            values: Values used to create the current parameter.

        Returns:
            The validated dictionary containing the values used to create the current
                parameters.
        """
        for value in values["default"]:
            if value not in values["options"]:
                raise ValueError(
                    f"Default value `{value}` is not available in the options."
                )

        if len(values["default"]) > 1 and not values["is_multi"]:
            raise ValueError(
                "`is_multi` is set to False, therefore only one default value should be given."
            )

        return values

    @classmethod
    def _get_option_type(cls) -> Type:
        """Returns the type of the `OptionT`.

        Returns:
            `OptionT` type.
        """
        # Trick to get the generic type, impossible to use traditional method, see:
        # https://github.com/pydantic/pydantic/issues/3559
        return get_type_hints(get_args(get_type_hints(cls)["options"])[1])["value"]

    def get_available_options(self) -> Set[OptionT]:
        """Returns available options.

        Returns:
            Available options.
        """
        # We use set here for faster search
        return {option.value for option in self.options.values()}

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
        """Process a value using the current parameter.

        Args:
            value: Value to process.

        Returns:
            Processed value.
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
        """Return default value for the parameter.

        Returns:
            Default value.
        """
        if self.dict_parameter:
            default = {option_name: {} for option_name in self.default}
        else:
            default = [self.options[option_name].value for option_name in self.default]

        return self.process_parameter(default)


class OptionsSpace(OptionsParameter[OptionT], Generic[OptionT]):
    """Parameter representing an options space."""

    is_multi: bool = True
