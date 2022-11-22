import pytest

from actableai.parameters.type import ValueType
from actableai.parameters.validation import ParameterTypeError
from actableai.parameters.value import ValueParameter


class TestValueParameter:
    @staticmethod
    def _create_parameter(parameter_type, default):
        return ValueParameter[parameter_type](
            name="value_parameter", display_name="Value Parameter", default=default
        )

    @pytest.mark.parametrize(
        "parameter_type,expected_value_type,default",
        [
            (int, ValueType.INT, 1),
            (float, ValueType.FLOAT, 1.5),
            (bool, ValueType.BOOL, True),
        ],
    )
    def test_set_value_type(self, parameter_type, expected_value_type, default):
        parameter = self._create_parameter(
            parameter_type=parameter_type,
            default=default,
        )

        assert parameter.value_type == expected_value_type

    @pytest.mark.parametrize(
        "parameter_type,default,value",
        [
            (int, 1, 1),
            (float, 1.5, 1.5),
            (float, 1.5, 1),
            (bool, True, True),
        ],
    )
    def test_validate_parameter_valid(self, parameter_type, default, value):
        parameter = self._create_parameter(
            parameter_type=parameter_type,
            default=default,
        )
        validation_errors = parameter.validate_parameter(value)

        assert len(validation_errors) == 0

    @pytest.mark.parametrize(
        "parameter_type,default,value",
        [
            (int, 1, "invalid"),
            (float, 1.5, "invalid"),
            (bool, True, "invalid"),
        ],
    )
    def test_validate_parameter_invalid(self, parameter_type, default, value):
        parameter = self._create_parameter(
            parameter_type=parameter_type,
            default=default,
        )
        validation_errors = parameter.validate_parameter(value)

        assert len(validation_errors) == 1
        assert validation_errors.has_error(ParameterTypeError)
