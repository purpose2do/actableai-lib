from typing import Any

import pytest

from actableai.parameters.base import BaseParameter, ProcessableParameter
from actableai.parameters.type import ParameterType
from actableai.parameters.validation import (
    ParameterTypeError,
    ParameterValidationError,
    ParameterValidationErrors,
)


@pytest.fixture(scope="module")
def bool_parameter():
    yield BaseParameter(
        name="bool_parameter",
        display_name="Bool Parameter",
        parameter_type=ParameterType.BOOL,
        default=True,
    )


@pytest.fixture(scope="module")
def int_parameter():
    yield BaseParameter(
        name="int_parameter",
        display_name="Int Parameter",
        parameter_type=ParameterType.INT,
        default=0,
    )


@pytest.fixture(scope="module")
def float_parameter():
    yield BaseParameter(
        name="float_parameter",
        display_name="Float Parameter",
        parameter_type=ParameterType.FLOAT,
        default=0.1,
    )


@pytest.fixture(scope="module")
def int_range_parameter():
    yield BaseParameter(
        name="int_range_parameter",
        display_name="Int Range Parameter",
        parameter_type=ParameterType.INT_RANGE,
        default=(0, 10),
    )


@pytest.fixture(scope="module")
def float_range_parameter():
    yield BaseParameter(
        name="float_range_parameter",
        display_name="Float Range Parameter",
        parameter_type=ParameterType.FLOAT_RANGE,
        default=(0.5, 1.5),
    )


@pytest.fixture(scope="module")
def option_parameter():
    yield BaseParameter(
        name="option_parameter",
        display_name="Option Parameter",
        parameter_type=ParameterType.OPTIONS,
        default="default",
    )


class TestBaseParameter:
    def test_validate_bool_parameter_valid(self, bool_parameter):
        validation_errors = bool_parameter.validate_parameter(True)

        assert len(validation_errors) == 0

    def test_validate_bool_parameter_invalid(self, bool_parameter):
        validation_errors = bool_parameter.validate_parameter("invalid")

        assert len(validation_errors) == 1
        assert validation_errors.has_error(ParameterTypeError)

    def test_validate_int_parameter_valid(self, int_parameter):
        validation_errors = int_parameter.validate_parameter(0)

        assert len(validation_errors) == 0

    def test_validate_int_parameter_invalid(self, int_parameter):
        validation_errors = int_parameter.validate_parameter("invalid")

        assert len(validation_errors) == 1
        assert validation_errors.has_error(ParameterTypeError)

    def test_validate_float_parameter_valid(self, float_parameter):
        validation_errors = float_parameter.validate_parameter(0.1)

        assert len(validation_errors) == 0

    def test_validate_float_parameter_invalid(self, float_parameter):
        validation_errors = float_parameter.validate_parameter("invalid")

        assert len(validation_errors) == 1
        assert validation_errors.has_error(ParameterTypeError)

    @pytest.mark.parametrize(
        "value",
        [
            [0, 10],
            (0, 10),
        ],
    )
    def test_validate_int_range_parameter_valid(self, int_range_parameter, value):
        validation_errors = int_range_parameter.validate_parameter(value)

        assert len(validation_errors) == 0

    @pytest.mark.parametrize(
        "value",
        [
            "invalid",
            ["invalid", "invalid2"],
            ("invalid", "invalid2"),
        ],
    )
    def test_validate_int_range_parameter_invalid_type(
        self, int_range_parameter, value
    ):
        validation_errors = int_range_parameter.validate_parameter(value)

        assert len(validation_errors) == 1
        assert validation_errors.has_error(ParameterTypeError)

    @pytest.mark.parametrize(
        "value",
        [
            [0, 10, 20],
            [],
            (0, 10, 20),
        ],
    )
    def test_validate_int_range_parameter_invalid_size(
        self, int_range_parameter, value
    ):
        validation_errors = int_range_parameter.validate_parameter(value)

        assert len(validation_errors) == 1
        assert validation_errors.has_error(ParameterValidationError)

    @pytest.mark.parametrize(
        "value",
        [
            [0.5, 10.5],
            (0.5, 10.5),
        ],
    )
    def test_validate_float_range_parameter_valid(self, float_range_parameter, value):
        validation_errors = float_range_parameter.validate_parameter(value)

        assert len(validation_errors) == 0

    @pytest.mark.parametrize(
        "value",
        [
            "invalid",
            ["invalid", "invalid2"],
            ("invalid", "invalid2"),
        ],
    )
    def test_validate_float_range_parameter_invalid_type(
        self, float_range_parameter, value
    ):
        validation_errors = float_range_parameter.validate_parameter(value)

        assert len(validation_errors) == 1
        assert validation_errors.has_error(ParameterTypeError)

    @pytest.mark.parametrize(
        "value",
        [
            [0.5, 10.5, 20.5],
            [],
            (0.5, 10.5, 20.5),
        ],
    )
    def test_validate_float_range_parameter_invalid_size(
        self, float_range_parameter, value
    ):
        validation_errors = float_range_parameter.validate_parameter(value)

        assert len(validation_errors) == 1
        assert validation_errors.has_error(ParameterValidationError)

    def test_validate_options_valid(self, option_parameter):
        validation_errors = option_parameter.validate_parameter("valid")

        assert len(validation_errors) == 0


class TestProcessableParameter:
    @staticmethod
    def _get_mock_parameter(valid: bool):
        class MockProcessableParameter(ProcessableParameter):
            def process_parameter(self, value: int) -> int:
                return value + 1

            def validate_parameter(self, value: Any) -> ParameterValidationErrors:
                errors = ParameterValidationErrors(parameter_name="test_parameter")
                if valid:
                    return errors

                errors.add_error(ParameterValidationError(parameter_name="test_error"))
                return errors

        return MockProcessableParameter()

    def test_validate_process_valid(self):
        parameter = self._get_mock_parameter(valid=True)
        errors, value = parameter.validate_process_parameter(0)

        assert len(errors) == 0
        assert value == 1

    def test_validate_process_invalid(self):
        parameter = self._get_mock_parameter(valid=False)
        errors, value = parameter.validate_process_parameter("value")

        assert len(errors) == 1
        assert value is None
