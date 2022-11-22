import pytest

from actableai.parameters.list import ListParameter
from actableai.parameters.type import ValueType
from actableai.parameters.validation import OutOfRangeError, ParameterTypeError


class TestListParameter:
    @staticmethod
    def _create_parameter(parameter_type, default, min_len=0, max_len=10):
        return ListParameter[parameter_type](
            name="list_parameter",
            display_name="List Parameter",
            default=default,
            min_len=min_len,
            max_len=max_len,
        )

    @pytest.mark.parametrize(
        "default",
        [
            1,
            (1, 2),
            [1, 2],
        ],
    )
    def test_set_default(self, default):
        parameter = self._create_parameter(
            parameter_type=int,
            default=default,
        )

        default = parameter.default
        assert isinstance(default, list)

    @pytest.mark.parametrize(
        "parameter_type,expected_value_type,default",
        [
            (int, ValueType.INT, [1, 2]),
            (float, ValueType.FLOAT, [1.5, 2.5]),
            (bool, ValueType.BOOL, [True, False]),
        ],
    )
    def test_set_value_type(self, parameter_type, expected_value_type, default):
        parameter = self._create_parameter(
            parameter_type=parameter_type,
            default=default,
        )

        assert parameter.value_type == expected_value_type

    def test_check_min_len_valid(self):
        self._create_parameter(
            parameter_type=int,
            default=[1, 2],
            min_len=0,
            max_len=10,
        )

    def test_check_min_len_invalid(self):
        with pytest.raises(ValueError):
            self._create_parameter(
                parameter_type=int,
                default=[1, 2],
                min_len=-1,
                max_len=10,
            )

    def test_check_len_valid(self):
        self._create_parameter(
            parameter_type=int,
            default=[1, 2],
            min_len=0,
            max_len=10,
        )

    def test_check_len_invalid(self):
        with pytest.raises(ValueError):
            self._create_parameter(
                parameter_type=int,
                default=[1, 2],
                min_len=5,
                max_len=1,
            )

    def test_check_default_len_valid(self):
        self._create_parameter(
            parameter_type=int,
            default=[1, 2],
            min_len=0,
            max_len=10,
        )

    @pytest.mark.parametrize(
        "min_len,max_len,default",
        [
            (1, 3, [1, 2, 3, 4, 5]),
            (3, 5, [1, 2]),
        ],
    )
    def test_check_default_len_invalid(self, min_len, max_len, default):
        with pytest.raises(ValueError):
            self._create_parameter(
                parameter_type=int,
                default=default,
                min_len=min_len,
                max_len=max_len,
            )

    @pytest.mark.parametrize(
        "parameter_type,default,value",
        [
            (int, [1, 2], [1, 2]),
            (int, [1, 2], (1, 2)),
            (int, [1, 2], 1),
            (float, [1.5, 2.5], [1.5, 2.5]),
            (float, [1.5, 2.5], (1.5, 2.5)),
            (float, [1.5, 2.5], 1.5),
            (float, [1.5, 2.5], [1, 2]),
            (float, [1.5, 2.5], (1, 2)),
            (float, [1.5, 2.5], 1),
            (bool, [True, False], [True, False]),
            (bool, [True, False], (True, False)),
            (bool, [True, False], True),
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
        "default,min_len,max_len,value",
        [
            ([1, 2, 4], 3, 10, [1, 2]),
            ([1, 2, 4], 3, 10, (1, 2)),
            ([1, 2], 0, 3, [1, 2, 3, 4]),
            ([1, 2], 0, 3, (1, 2, 3, 4)),
        ],
    )
    def test_validate_parameter_invalid_size(self, default, min_len, max_len, value):
        parameter = self._create_parameter(
            parameter_type=int,
            default=default,
            min_len=min_len,
            max_len=max_len,
        )
        validation_error = parameter.validate_parameter(value)

        assert len(validation_error) == 1
        assert validation_error.has_error(OutOfRangeError)

    @pytest.mark.parametrize(
        "parameter_type,default,value",
        [
            (int, [1, 2], ["invalid", "invalid_too"]),
            (int, [1, 2], [1, "invalid"]),
            (float, [1.5, 2.5], ["invalid", "invalid_too"]),
            (float, [1.5, 2.5], [1.5, "invalid"]),
            (bool, [True, False], ["invalid", "invalid_too"]),
            (bool, [True, False], [True, "invalid_too"]),
        ],
    )
    def test_validate_parameter_invalid_type(self, parameter_type, default, value):
        parameter = self._create_parameter(
            parameter_type=parameter_type,
            default=default,
        )
        validation_errors = parameter.validate_parameter(value)

        assert len(validation_errors) >= 1
        assert validation_errors.has_error(ParameterTypeError)

    @pytest.mark.parametrize(
        "value",
        [
            1,
            (1, 2),
            [1, 2],
        ],
    )
    def test_process_parameter(self, value):
        parameter = self._create_parameter(
            parameter_type=int,
            default=[1, 2],
        )
        processed_value = parameter.process_parameter(value)

        assert isinstance(processed_value, list)
