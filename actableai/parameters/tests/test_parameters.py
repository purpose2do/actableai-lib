import pytest

from actableai.parameters.parameters import Parameters
from actableai.parameters.validation import ParameterTypeError, InvalidKeyError
from actableai.parameters.value import ValueParameter


class TestParameters:
    @staticmethod
    def _create_parameters(parameters=None):
        if parameters is None:
            parameters = [
                ValueParameter[bool](
                    name="param_1",
                    display_name="Param 1",
                    default=True,
                ),
                ValueParameter[bool](
                    name="param_2",
                    display_name="Param 2",
                    default=True,
                ),
            ]

        return Parameters(
            name="parameters",
            display_name="Parameters",
            parameters=parameters,
        )

    @pytest.mark.parametrize(
        "available_parameters",
        [
            [
                ValueParameter[bool](
                    name="param_1",
                    display_name="Param 1",
                    default=True,
                ),
                ValueParameter[bool](
                    name="param_2",
                    display_name="Param 2",
                    default=True,
                ),
            ],
            {
                "param_1": ValueParameter[bool](
                    name="param_1",
                    display_name="Param 1",
                    default=True,
                ),
                "param_2": ValueParameter[bool](
                    name="param_2",
                    display_name="Param 2",
                    default=True,
                ),
            },
        ],
    )
    def test_set_parameters(self, available_parameters):
        parameters = self._create_parameters(parameters=available_parameters)

        assert isinstance(parameters.parameters, dict)
        assert "param_1" in parameters.parameters
        assert "param_2" in parameters.parameters

    @pytest.mark.parametrize(
        "selected_parameters",
        [
            {},
            {
                "param_1": True,
            },
            {
                "param_1": True,
                "param_2": True,
            },
        ],
    )
    def test_validate_valid(self, selected_parameters):
        parameters = self._create_parameters()
        validation_errors = parameters.validate_parameter(selected_parameters)

        assert len(validation_errors) == 0

    @pytest.mark.parametrize(
        "selected_parameters",
        [
            True,
            [True],
        ],
    )
    def test_validate_type_invalid(self, selected_parameters):
        parameters = self._create_parameters()
        validation_errors = parameters.validate_parameter(selected_parameters)

        assert len(validation_errors) == 1
        assert validation_errors.has_error(ParameterTypeError)

    @pytest.mark.parametrize(
        "selected_parameters",
        [
            {
                "param_3": True,
            },
            {
                "param_2": True,
                "param_3": True,
            },
            {
                "param_3": True,
                "param_4": True,
            },
        ],
    )
    def test_validate_key_invalid(self, selected_parameters):
        parameters = self._create_parameters()
        validation_errors = parameters.validate_parameter(selected_parameters)

        assert len(validation_errors) >= 1
        assert validation_errors.has_error(InvalidKeyError)

    @pytest.mark.parametrize(
        "selected_parameters",
        [
            {
                "param_1": "invalid",
            },
            {
                "param_1": True,
                "param_2": "invalid",
            },
            {
                "param_1": "invalid",
                "param_2": "invalid",
            },
        ],
    )
    def test_validate_invalid_sub(self, selected_parameters):
        parameters = self._create_parameters()
        validation_errors = parameters.validate_parameter(selected_parameters)

        assert len(validation_errors) >= 1

    @pytest.mark.parametrize(
        "selected_parameters",
        [
            {},
            {
                "param_1": 0,
            },
            {
                "param_1": 0,
                "param_2": 0,
            },
        ],
    )
    def test_process(self, selected_parameters):
        class MockParameter(ValueParameter[int]):
            default: int = 0

            def process_parameter(self, v: int) -> int:
                return v + 1

        param_1 = MockParameter(
            name="param_1",
            display_name="Param 1",
        )

        param_2 = MockParameter(
            name="param_2",
            display_name="Param 2",
        )

        parameters = Parameters(
            name="parameters",
            display_name="Parameters",
            parameters=[param_1, param_2],
        )
        processed_parameters = parameters.process_parameter(selected_parameters)

        assert len(processed_parameters) == 2
        assert "param_1" in processed_parameters
        assert "param_2" in processed_parameters

        assert processed_parameters["param_1"] == 1
        assert processed_parameters["param_2"] == 1
