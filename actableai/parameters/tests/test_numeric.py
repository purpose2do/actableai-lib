import pytest

from actableai.parameters.numeric import NumericParameter, NumericRangeSpace
from actableai.parameters.type import ParameterType
from actableai.parameters.validation import OutOfRangeError


class TestNumericParameter:
    @staticmethod
    def _create_parameter(numeric_type, default, min_value, max_value):
        parameter_type = (
            ParameterType.INT if numeric_type == int else ParameterType.FLOAT
        )

        return NumericParameter[numeric_type](
            name="numeric_parameter",
            display_name="Numeric Parameter",
            parameter_type=parameter_type,
            default=default,
            min=min_value,
            max=max_value,
        )

    @pytest.mark.parametrize("numeric_type", [int, float])
    @pytest.mark.parametrize(
        "min_value,max_value",
        [
            [None, None],
            [-5, None],
            [None, 5],
            [-5, 5],
        ],
    )
    def test_check_default_valid(self, numeric_type, min_value, max_value):
        self._create_parameter(
            numeric_type=numeric_type,
            default=0,
            min_value=min_value,
            max_value=max_value,
        )

    @pytest.mark.parametrize("numeric_type", [int, float])
    @pytest.mark.parametrize(
        "min_value,max_value,value",
        [
            [-5, None, -10],
            [None, 5, 10],
            [-5, 5, -10],
            [-5, 5, 10],
        ],
    )
    def test_check_default_invalid(self, numeric_type, min_value, max_value, value):
        with pytest.raises(ValueError):
            self._create_parameter(
                numeric_type=numeric_type,
                default=value,
                min_value=min_value,
                max_value=max_value,
            )

    @pytest.mark.parametrize("numeric_type", [int, float])
    @pytest.mark.parametrize(
        "min_value,max_value",
        [
            [None, None],
            [-5, None],
            [None, 5],
            [-5, 5],
        ],
    )
    def test_validate_valid(self, numeric_type, min_value, max_value):
        numeric_parameter = self._create_parameter(
            numeric_type=numeric_type,
            default=0,
            min_value=min_value,
            max_value=max_value,
        )
        validation_errors = numeric_parameter.validate_parameter(0)

        assert len(validation_errors) == 0

    @pytest.mark.parametrize("numeric_type", [int, float])
    @pytest.mark.parametrize(
        "min_value,max_value,value",
        [
            [-5, None, -10],
            [None, 5, 10],
            [-5, 5, -10],
            [-5, 5, 10],
        ],
    )
    def test_validate_invalid(self, numeric_type, min_value, max_value, value):
        numeric_parameter = self._create_parameter(
            numeric_type=numeric_type,
            default=0,
            min_value=min_value,
            max_value=max_value,
        )
        validation_errors = numeric_parameter.validate_parameter(value)

        assert len(validation_errors) == 1
        assert validation_errors.has_error(OutOfRangeError)


class TestNumericRangeSpace:
    @staticmethod
    def _create_parameter(numeric_type, default, min_value, max_value):
        parameter_type = (
            ParameterType.INT_RANGE
            if numeric_type == int
            else ParameterType.FLOAT_RANGE
        )

        return NumericRangeSpace[numeric_type](
            name="numeric_range_space",
            display_name="Numeric Range Space",
            parameter_type=parameter_type,
            default=default,
            min=min_value,
            max=max_value,
        )

    @pytest.mark.parametrize("numeric_type", [int, float])
    @pytest.mark.parametrize(
        "min_value,max_value",
        [
            [None, None],
            [-10, None],
            [None, 10],
            [-10, 10],
        ],
    )
    @pytest.mark.parametrize(
        "value",
        [
            0,
            (-5, 5),
        ],
    )
    def test_check_default_valid(self, numeric_type, min_value, max_value, value):
        self._create_parameter(
            numeric_type=numeric_type,
            default=value,
            min_value=min_value,
            max_value=max_value,
        )

    @pytest.mark.parametrize("numeric_type", [int, float])
    @pytest.mark.parametrize(
        "value",
        [
            -10,
            (-10, 10),
        ],
    )
    def test_check_default_lower_limit_invalid(self, numeric_type, value):
        with pytest.raises(ValueError):
            self._create_parameter(
                numeric_type=numeric_type,
                default=value,
                min_value=-5,
                max_value=None,
            )

    @pytest.mark.parametrize("numeric_type", [int, float])
    @pytest.mark.parametrize(
        "value",
        [
            10,
            (-10, 10),
        ],
    )
    def test_check_default_upper_limit_invalid(self, numeric_type, value):
        with pytest.raises(ValueError):
            self._create_parameter(
                numeric_type=numeric_type,
                default=value,
                min_value=None,
                max_value=5,
            )

    @pytest.mark.parametrize("numeric_type", [int, float])
    @pytest.mark.parametrize(
        "value",
        [
            -10,
            10,
            (-10, 0),
            (-10, 10),
            (0, 10),
        ],
    )
    def test_check_default_lower_upper_limit_invalid(self, numeric_type, value):
        with pytest.raises(ValueError):
            self._create_parameter(
                numeric_type=numeric_type,
                default=value,
                min_value=-5,
                max_value=5,
            )

    @pytest.mark.parametrize("numeric_type", [int, float])
    @pytest.mark.parametrize(
        "min_value,max_value",
        [
            [None, None],
            [-10, None],
            [None, 10],
            [-10, 10],
        ],
    )
    @pytest.mark.parametrize(
        "value",
        [
            0,
            [0],
            (0,),
            [-5, 5],
            (-5, 5),
        ],
    )
    def test_validate_valid(self, numeric_type, min_value, max_value, value):
        numeric_parameter = self._create_parameter(
            numeric_type=numeric_type,
            default=(-1, 1),
            min_value=min_value,
            max_value=max_value,
        )
        validation_errors = numeric_parameter.validate_parameter(value)

        assert len(validation_errors) == 0

    @pytest.mark.parametrize("numeric_type", [int, float])
    @pytest.mark.parametrize(
        "value",
        [
            -10,
            [-10],
            (-10,),
            [-10, 10],
            (-10, 10),
        ],
    )
    def test_validate_lower_limit_invalid(self, numeric_type, value):
        numeric_parameter = self._create_parameter(
            numeric_type=numeric_type,
            default=(-1, 1),
            min_value=-5,
            max_value=None,
        )
        validation_errors = numeric_parameter.validate_parameter(value)

        assert len(validation_errors) == 1
        assert validation_errors.has_error(OutOfRangeError)

    @pytest.mark.parametrize("numeric_type", [int, float])
    @pytest.mark.parametrize(
        "value",
        [
            10,
            [10],
            (10,),
            [-10, 10],
            (-10, 10),
        ],
    )
    def test_validate_upper_limit_invalid(self, numeric_type, value):
        numeric_parameter = self._create_parameter(
            numeric_type=numeric_type,
            default=(-1, 1),
            min_value=None,
            max_value=5,
        )
        validation_errors = numeric_parameter.validate_parameter(value)

        assert len(validation_errors) == 1
        assert validation_errors.has_error(OutOfRangeError)

    @pytest.mark.parametrize("numeric_type", [int, float])
    @pytest.mark.parametrize(
        "value",
        [
            -10,
            10,
            [-10],
            [10],
            (-10,),
            (10,),
            [-10, 0],
            [-10, 10],
            [0, 10],
            (-10, 0),
            (-10, 10),
            (0, 10),
        ],
    )
    def test_validate_lower_upper_limit_invalid(self, numeric_type, value):
        numeric_parameter = self._create_parameter(
            numeric_type=numeric_type,
            default=(-1, 1),
            min_value=-5,
            max_value=5,
        )
        validation_errors = numeric_parameter.validate_parameter(value)

        assert len(validation_errors) >= 1
        assert validation_errors.has_error(OutOfRangeError)

    @pytest.mark.parametrize("numeric_type", [int, float])
    @pytest.mark.parametrize(
        "min_value,max_value",
        [
            [None, None],
            [-10, None],
            [None, 10],
            [-10, 10],
        ],
    )
    @pytest.mark.parametrize(
        "value",
        [
            0,
            [0],
            (0,),
            [-5, 5],
            (-5, 5),
        ],
    )
    def test_process(self, numeric_type, min_value, max_value, value):
        numeric_parameter = self._create_parameter(
            numeric_type=numeric_type,
            default=(-1, 1),
            min_value=min_value,
            max_value=max_value,
        )
        processed_value = numeric_parameter.process_parameter(value)

        if not isinstance(value, (tuple, list)) or len(value) == 1:
            assert isinstance(processed_value, (int, float))
            processed_value = [processed_value]
        else:
            assert isinstance(processed_value, tuple)

        for val in processed_value:
            assert isinstance(val, (int, float))
