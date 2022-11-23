from typing import Any

from actableai.parameters.base import BaseParameter, ProcessableParameter
from actableai.parameters.type import ParameterType
from actableai.parameters.validation import (
    ParameterValidationError,
    ParameterValidationErrors,
)


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


class TestBaseParameter:
    def test_get_default(self):
        class MockBaseParameter(BaseParameter):
            def process_parameter(self, value: int) -> int:
                return value + 1

        parameter = MockBaseParameter(
            name="test_parameter",
            display_name="Test Parameter",
            parameter_type=ParameterType.VALUE,
            default=0,
        )

        default = parameter.get_default()
        assert default == 1
