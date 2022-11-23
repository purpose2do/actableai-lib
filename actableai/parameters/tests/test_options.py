from typing import Any, Dict, Optional

import pytest

from actableai.parameters.boolean import BooleanParameter
from actableai.parameters.options import OptionsParameter
from actableai.parameters.validation import (
    ParameterTypeError,
    ParameterValidationError,
    InvalidKeyError,
)
from actableai.parameters.value import ValueParameter


class TestOptionsParameter:
    @staticmethod
    def _create_parameter(
        parameter_type, is_multi, default, options, dict_parameter=False
    ):
        return OptionsParameter[parameter_type](
            name="options_parameter",
            display_name="Options Parameter",
            dict_parameter=dict_parameter,
            is_multi=is_multi,
            default=default,
            options=options,
        )

    @classmethod
    def _create_bool_parameter_parameter(
        cls, is_multi: Any, default: Any = "bool_1", dict_parameter: Any = False
    ):
        return cls._create_parameter(
            parameter_type=BooleanParameter,
            is_multi=is_multi,
            default=default,
            options={
                "bool_1": {
                    "display_name": "Bool 1",
                    "value": {
                        "name": "bool_parameter_1",
                        "display_name": "Bool Parameter 1",
                        "default": True,
                    },
                },
                "bool_2": {
                    "display_name": "Bool 2",
                    "value": {
                        "name": "bool_parameter_2",
                        "display_name": "Bool Parameter 2",
                        "default": False,
                    },
                },
            },
            dict_parameter=dict_parameter,
        )

    @classmethod
    def _create_int_parameter(
        cls,
        is_multi: Any,
        default: Any = "option_1",
        dict_parameter: Any = False,
    ):
        return cls._create_parameter(
            parameter_type=int,
            is_multi=is_multi,
            default=default,
            options={
                "option_1": {"display_name": "Option 1", "value": 1},
                "option_2": {"display_name": "Option 2", "value": 2},
            },
        )

    @pytest.mark.parametrize("dict_parameter", [True, False])
    @pytest.mark.parametrize("is_multi", [True, False])
    def test_set_dict_parameter_true(self, dict_parameter, is_multi):
        options_parameter = self._create_bool_parameter_parameter(
            is_multi=is_multi,
            dict_parameter=dict_parameter,
        )

        assert options_parameter.dict_parameter is True

    @pytest.mark.parametrize("dict_parameter", [True, False])
    @pytest.mark.parametrize("is_multi", [True, False])
    def test_set_dict_parameter_false(self, dict_parameter, is_multi):
        options_parameter = self._create_int_parameter(
            is_multi=is_multi,
            dict_parameter=dict_parameter,
        )

        assert options_parameter.dict_parameter is False

    @pytest.mark.parametrize(
        "is_multi,default",
        [
            [True, "option_1"],
            [True, ["option_1"]],
            [True, ["option_1", "option_2"]],
            [False, "option_1"],
            [False, ["option_1"]],
        ],
    )
    def test_set_default(self, is_multi, default):
        options_parameter = self._create_int_parameter(
            is_multi=is_multi,
            default=default,
        )

        assert isinstance(options_parameter.default, list)

    @pytest.mark.parametrize(
        "is_multi,default",
        [
            [True, "option_1"],
            [True, ["option_1"]],
            [True, ["option_1", "option_2"]],
            [False, "option_1"],
            [False, ["option_1"]],
        ],
    )
    def test_check_default_valid(self, is_multi, default):
        self._create_int_parameter(
            is_multi=is_multi,
            default=default,
        )

    @pytest.mark.parametrize("is_multi", [True, False])
    @pytest.mark.parametrize(
        "default",
        [
            "invalid_option",
            ["invalid_option"],
            ["option_1", "invalid_option"],
        ],
    )
    def test_check_default_option_invalid(self, is_multi, default):
        with pytest.raises(ValueError):
            self._create_int_parameter(
                is_multi=is_multi,
                default=default,
            )

    def test_check_default_option_length_invalid(self):
        with pytest.raises(ValueError):
            self._create_int_parameter(
                is_multi=False,
                default=["option_1", "option_2"],
            )

    @pytest.mark.parametrize("is_multi", [True, False])
    def test_available_options(self, is_multi):
        options_parameter = self._create_int_parameter(is_multi=is_multi)
        available_options = options_parameter.get_available_options()

        assert 1 in available_options
        assert 2 in available_options

    @pytest.mark.parametrize(
        "is_multi,value",
        [
            [True, 1],
            [True, [1]],
            [True, (1,)],
            [True, [1, 2]],
            [True, (1, 2)],
            [False, 1],
            [False, [1]],
            [False, (1,)],
        ],
    )
    def test_validate_simple_valid(self, is_multi, value):
        options_parameter = self._create_int_parameter(is_multi=is_multi)
        validation_errors = options_parameter.validate_parameter(value)

        assert len(validation_errors) == 0

    @pytest.mark.parametrize(
        "is_multi,value",
        [
            [True, {"bool_1": True}],
            [True, {"bool_1": True, "bool_2": True}],
            [False, {"bool_1": True}],
        ],
    )
    def test_validate_complex_valid(self, is_multi, value):
        options_parameter = self._create_bool_parameter_parameter(is_multi=is_multi)
        validation_errors = options_parameter.validate_parameter(value)

        assert len(validation_errors) == 0

    @pytest.mark.parametrize(
        "is_multi,value",
        [
            [True, {"option_1": 1}],
            [True, {"option_1": 1, "option_2": 2}],
            [False, {"option_1": 1}],
        ],
    )
    def test_validate_simple_type_invalid(self, is_multi, value):
        options_parameter = self._create_int_parameter(is_multi=is_multi)
        validation_errors = options_parameter.validate_parameter(value)

        assert len(validation_errors) == 1
        assert validation_errors.has_error(ParameterTypeError)

    @pytest.mark.parametrize(
        "is_multi,value",
        [
            [True, True],
            [True, [True]],
            [True, (True,)],
            [True, [True, False]],
            [True, (True, False)],
            [False, True],
            [False, [True]],
            [False, (True,)],
        ],
    )
    def test_validate_complex_type_invalid(self, is_multi, value):
        options_parameter = self._create_bool_parameter_parameter(is_multi=is_multi)
        validation_errors = options_parameter.validate_parameter(value)

        assert len(validation_errors) == 1
        assert validation_errors.has_error(ParameterTypeError)

    @pytest.mark.parametrize(
        "is_multi,value",
        [
            [True, []],
            [False, []],
            [False, [1, 2]],
            [False, (1, 2)],
        ],
    )
    def test_validate_simple_length_invalid(self, is_multi, value):
        options_parameter = self._create_int_parameter(is_multi=is_multi)
        validation_errors = options_parameter.validate_parameter(value)

        assert len(validation_errors) == 1
        assert validation_errors.has_error(ParameterValidationError)

    @pytest.mark.parametrize(
        "is_multi,value",
        [
            [True, {}],
            [False, {}],
            [False, {"bool_1": True, "bool_2": True}],
        ],
    )
    def test_validate_complex_length_invalid(self, is_multi, value):
        options_parameter = self._create_bool_parameter_parameter(is_multi=is_multi)
        validation_errors = options_parameter.validate_parameter(value)

        assert len(validation_errors) == 1
        assert validation_errors.has_error(ParameterValidationError)

    @pytest.mark.parametrize(
        "is_multi,value",
        [
            [True, 3],
            [True, [3]],
            [True, (3,)],
            [True, [2, 3]],
            [True, (2, 3)],
            [True, [3, 4]],
            [True, (3, 4)],
            [False, 3],
            [False, [3]],
            [False, (3,)],
        ],
    )
    def test_validate_simple_key_invalid(self, is_multi, value):
        options_parameter = self._create_int_parameter(is_multi=is_multi)
        validation_errors = options_parameter.validate_parameter(value)

        assert len(validation_errors) >= 1
        assert validation_errors.has_error(InvalidKeyError)

    @pytest.mark.parametrize(
        "is_multi,value",
        [
            [True, {"bool_3": True}],
            [True, {"bool_2": True, "bool_3": True}],
            [True, {"bool_3": True, "bool_4": True}],
            [False, {"bool_3": True}],
        ],
    )
    def test_validate_complex_key_invalid(self, is_multi, value):
        options_parameter = self._create_bool_parameter_parameter(is_multi=is_multi)
        validation_errors = options_parameter.validate_parameter(value)

        assert len(validation_errors) >= 1
        assert validation_errors.has_error(InvalidKeyError)

    @pytest.mark.parametrize(
        "is_multi,value",
        [
            [True, {"bool_1": "invalid"}],
            [True, {"bool_1": "invalid", "bool_2": "invalid"}],
            [False, {"bool_1": "invalid"}],
        ],
    )
    def test_validate_complex_invalid_sub(self, is_multi, value):
        options_parameter = self._create_bool_parameter_parameter(is_multi=is_multi)
        validation_errors = options_parameter.validate_parameter(value)

        assert len(validation_errors) >= 1

    @pytest.mark.parametrize(
        "is_multi,value",
        [
            [True, 1],
            [True, [1]],
            [True, (1,)],
            [True, [1, 2]],
            [True, (1, 2)],
            [False, 1],
            [False, [1]],
            [False, (1,)],
        ],
    )
    def test_process_simple(self, is_multi, value):
        options_parameter = self._create_int_parameter(is_multi=is_multi)
        processed_value = options_parameter.process_parameter(value)

        if not is_multi or isinstance(value, int) or len(value) == 1:
            assert isinstance(processed_value, int)
            processed_value = [processed_value]
        else:
            assert isinstance(processed_value, tuple)

        for val in processed_value:
            assert isinstance(val, int)

    @pytest.mark.parametrize(
        "is_multi,value",
        [
            [True, {"option_1": 0}],
            [True, {"option_1": 0, "option_2": 0}],
            [False, {"option_1": 0}],
        ],
    )
    def test_process_complex(self, is_multi: bool, value: Dict[str, int]):
        class MockParameter(ValueParameter[int]):
            default: Optional[int] = 0

            def process_parameter(self, v: int) -> int:
                return v + 1

        options_parameter = self._create_parameter(
            parameter_type=MockParameter,
            is_multi=is_multi,
            default="option_1",
            options={
                "option_1": {
                    "display_name": "Option 1",
                    "value": {
                        "name": "mock_parameter_1",
                        "display_name": "Mock Parameter 1",
                    },
                },
                "option_2": {
                    "display_name": "Option 2",
                    "value": {
                        "name": "mock_parameter_2",
                        "display_name": "Mock Parameter 2",
                    },
                },
            },
        )
        processed_value = options_parameter.process_parameter(value)

        for key, val in value.items():
            assert processed_value[key] == val + 1

    @pytest.mark.parametrize(
        "is_multi,default",
        [
            [True, "option_1"],
            [True, ["option_1"]],
            [True, ["option_1", "option_2"]],
            [False, "option_1"],
            [False, ["option_1"]],
        ],
    )
    def test_get_default_simple(self, is_multi, default):
        options_parameter = self._create_int_parameter(
            is_multi=is_multi,
            default=default,
        )
        default_processed = options_parameter.get_default()

        if not is_multi or isinstance(default, str) or len(default) == 1:
            assert isinstance(default_processed, int)
            default_processed = [default_processed]
        else:
            assert isinstance(default_processed, tuple)

        for val in default_processed:
            assert isinstance(val, int)

    @pytest.mark.parametrize(
        "is_multi,default",
        [
            [True, "bool_1"],
            [True, ["bool_1"]],
            [True, ["bool_1", "bool_2"]],
            [False, "bool_1"],
            [False, ["bool_1"]],
        ],
    )
    def test_get_default_complex(self, is_multi, default):
        options_parameter = self._create_bool_parameter_parameter(
            is_multi=is_multi,
            default=default,
        )
        default_processed = options_parameter.get_default()
        assert isinstance(default_processed, dict)

        if not isinstance(default, list):
            default = [default]

        for val in default:
            assert val in default_processed
