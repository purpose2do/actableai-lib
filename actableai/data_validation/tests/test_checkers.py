import numpy as np
import pandas as pd
import pytest

from actableai.data_validation.base import CheckLevels, CheckResult
from actableai.data_validation.checkers import (
    CategoricalSameValuesChecker,
    CheckColumnInflateLimit,
    CheckNUnique,
    DoNotContainMixedChecker,
    IsCategoricalChecker,
    IsCategoricalOrNumericalChecker,
    IsDatetimeChecker,
    IsNumericalChecker,
    IsSufficientClassSampleChecker,
    MaxTrainSamplesChecker,
    PositiveOutcomeValueThreshold,
    SameTypeChecker,
    StratifiedKFoldChecker,
)


class TestIsCategoricalChecker:
    def test_categorical_data(self):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": ["a", 2, 1, 2, 1, None, 1, 2, 1, 2],
                "z": ["1", "1", "1", "2", "2", "2", "3", "3", "3", "3"],
            }
        )
        c1 = IsCategoricalChecker(level=CheckLevels.CRITICAL).check(df["y"])  # type: ignore
        c2 = IsCategoricalChecker(level=CheckLevels.CRITICAL).check(df["z"])
        assert isinstance(c1, CheckResult)
        assert c2 is None


class TestDoNotContainMixedChecker:
    def test_mix_data(self):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": ["a", 2, 1, 2, 1, None, 1, 2, 1, 2],
                "z": ["1", "1", "1", "2", "2", "2", "3", "3", "3", "3"],
            }
        )
        c1 = DoNotContainMixedChecker(level=CheckLevels.CRITICAL).check(df, ["y"])
        c2 = DoNotContainMixedChecker(level=CheckLevels.CRITICAL).check(df, ["x"])
        assert isinstance(c1, CheckResult)
        assert c2 is None


class TestIsDatetimeChecker:
    def test_datetime_data(self):
        rng = pd.date_range("2015-02-24", periods=10, freq="T")
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": ["a", 2, 1, 2, 1, None, 1, 2, 1, 2],
                "z": ["1", "1", "1", "2", "2", "2", "3", "3", "3", "3"],
                "date": rng,
            }
        )
        c1 = IsDatetimeChecker(level=CheckLevels.CRITICAL).check(df["y"])
        c2 = IsDatetimeChecker(level=CheckLevels.CRITICAL).check(df["date"])  # type: ignore
        assert isinstance(c1, CheckResult)
        assert c2 is None


class TestCheckNUnique:
    @pytest.mark.parametrize("analytics", ["Explanation", "Bayesian Regression"])
    @pytest.mark.parametrize("n_unique_level", [10, 100])
    def test_check(self, analytics: str, n_unique_level: int):
        df = pd.DataFrame({"x": [str(x) for x in range(200)]})
        checker = CheckNUnique(level=CheckLevels.CRITICAL, name="CheckNUnique")
        checkresult = checker.check(
            df, n_unique_level=n_unique_level, analytics=analytics
        )

        assert checkresult is not None
        assert checkresult.name == "CheckNUnique"
        assert checkresult.level == CheckLevels.CRITICAL
        assert (
            checkresult.message
            == f"{analytics} currently doesn't support categorical columns with more than {n_unique_level} unique values.\n"
            + "['x'] column(s) have too many unique values."
        )

    @pytest.mark.parametrize("analytics", ["Explanation", "Bayesian Regression"])
    @pytest.mark.parametrize("n_unique_level", [10, 100])
    def test_check_mixed(self, analytics: str, n_unique_level: int):
        df = pd.DataFrame(
            {
                "x": [str(x) for x in range(200)],
                "y": [str(y) for y in range(200)],
                "z": ["a" for _ in range(200)],
            }
        )
        checker = CheckNUnique(level=CheckLevels.CRITICAL, name="CheckNUnique")
        checkresult = checker.check(
            df, n_unique_level=n_unique_level, analytics=analytics
        )

        assert checkresult is not None
        assert checkresult.name == "CheckNUnique"
        assert checkresult.level == CheckLevels.CRITICAL
        assert (
            checkresult.message
            == f"{analytics} currently doesn't support categorical columns with more than {n_unique_level} unique values.\n"
            + "['x', 'y'] column(s) have too many unique values."
        )


class TestIsNumericalChecker:
    @pytest.mark.parametrize(
        "series",
        [
            pd.Series([int(x) for x in range(10)]),
            pd.Series([float(x) for x in range(10)]),
        ],
    )
    def test_check(self, series: pd.Series):
        checker = IsNumericalChecker(
            level=CheckLevels.CRITICAL, name="IsNumericalChecker"
        )
        checkresult = checker.check(series)

        assert checkresult is None

    @pytest.mark.parametrize(
        "series, dtype",
        [
            [pd.Series([str(x) for x in range(10)]), "category"],
            [
                pd.Series(pd.date_range(start="27-03-1997", periods=10, freq="H")),
                "datetime",
            ],
        ],
    )
    def test_check_not(self, series: pd.Series, dtype: str):
        checker = IsNumericalChecker(
            level=CheckLevels.CRITICAL, name="IsNumericalChecker"
        )
        checkresult = checker.check(series)

        assert checkresult is not None
        assert checkresult.name == "IsNumericalChecker"
        assert checkresult.level == CheckLevels.CRITICAL
        assert (
            checkresult.message
            == f"Expected target '{series.name}' to be a numerical column, found {dtype} instead"
        )

    def test_numerical_data(self):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": ["a", 2, 1, 2, 1, None, 1, 2, 1, 2],
                "z": ["1", "1", "1", "2", "2", "2", "3", "3", "3", "3"],
            }
        )
        c1 = IsNumericalChecker(level=CheckLevels.CRITICAL).check(df["y"])
        c2 = IsNumericalChecker(level=CheckLevels.CRITICAL).check(df["x"])
        assert isinstance(c1, CheckResult)
        assert c2 is None


class TestCheckColumnInflateLimit:
    def test_check(self):
        checker = CheckColumnInflateLimit(
            level=CheckLevels.CRITICAL, name="CheckColumnInflateLimit"
        )
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        checkresult = checker.check(df, ["x"], 1, 2)

        assert checkresult is not None
        assert checkresult.name == "CheckColumnInflateLimit"
        assert checkresult.level == CheckLevels.CRITICAL
        assert (
            checkresult.message
            == "Dataset after inflation is too large. Please lower the polynomial degree or reduce the number of unique values in categorical columns."
        )

    def test_not_check(self):
        checker = CheckColumnInflateLimit(
            level=CheckLevels.CRITICAL, name="CheckColumnInflateLimit"
        )
        df = pd.DataFrame({"x": [1, 2, 3]})
        checkresult = checker.check(df, ["x"], 1, 2)

        assert checkresult is None


class TestIsSufficientClassSampleChecker:
    def test_check(self):
        iscsc = IsSufficientClassSampleChecker(
            level=CheckLevels.WARNING, name="IsSufficientClassSampleChecker"
        )
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d"],
            }
        )
        target = "y"
        validation_ratio = 0.2
        result = iscsc.check(df, target, validation_ratio)
        assert result is not None
        assert result.name == "IsSufficientClassSampleChecker"
        assert result.level == CheckLevels.CRITICAL


class TestMaxTrainSamplesChecker:
    def test_check(self):
        mtsc = MaxTrainSamplesChecker(
            level=CheckLevels.CRITICAL, name="MaxTrainSamplesChecker"
        )
        result = mtsc.check(n_cluster=3, max_samples=2)
        assert result is not None
        assert result.name == "MaxTrainSamplesChecker"
        assert result.level == CheckLevels.CRITICAL


class TestPositiveOutcomeValueThreshold:
    def test_check(self):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d"],
            }
        )

        result = PositiveOutcomeValueThreshold(
            level=CheckLevels.CRITICAL, name="PositiveOutcomeValueThreshold"
        ).check(df=df, outcomes=["x"], positive_outcome_value="tomato")
        assert result is not None
        assert result.name == "PositiveOutcomeValueThreshold"
        assert result.level == CheckLevels.CRITICAL


class TestIsCategoricalOrNumericalChecker:
    def test_check(self):
        checker = IsCategoricalOrNumericalChecker(
            level=CheckLevels.CRITICAL, name="IsCategoricalOrNumericalChecker"
        )

        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d"],
            }
        )

        result = checker.check(df, ["x", "y"])
        assert result is None


class TestSameTypeChecker:
    def test_check(self):
        checker = SameTypeChecker(level=CheckLevels.CRITICAL, name="SameTypeChecker")
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d"],
            }
        )
        result = checker.check(df, ["x", "y"])
        assert result is not None
        assert result.name == "SameTypeChecker"
        assert result.level == CheckLevels.CRITICAL


class TestCategoricalSameValuesChecker:
    def test_check(self):
        checker = CategoricalSameValuesChecker(
            level=CheckLevels.CRITICAL, name="CategoricalSameValuesChecker"
        )
        df = pd.DataFrame(
            {
                "x": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d"],
                "y": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "e"],
            }
        )
        result = checker.check(df, "x", "y")
        assert result is not None
        assert result.name == "CategoricalSameValuesChecker"
        assert result.level == CheckLevels.CRITICAL


class TestStratifiedKFoldChecker:
    def test_check(self):
        checker = StratifiedKFoldChecker(
            level=CheckLevels.CRITICAL, name="StratifiedKFoldChecker"
        )
        df = pd.DataFrame(
            {
                "x": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d"],
                "y": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "e"],
            }
        )
        result = checker.check(df, "x", 20)
        assert result is not None
        assert result.name == "StratifiedKFoldChecker"
        assert result.level == CheckLevels.CRITICAL
