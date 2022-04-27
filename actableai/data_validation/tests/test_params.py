import pandas as pd
from pandas._testing import rands_array
import numpy as np
from actableai.data_validation.base import CheckLevels

from actableai.data_validation.params import (
    BayesianRegressionDataValidator,
    CausalDataValidator,
    RegressionDataValidator,
    ClassificationDataValidator,
)


class TestBayesianRegressionDataValidator:
    def test_validate_CheckNUnique(self):
        df = pd.DataFrame(
            {
                "x": rands_array(10, 200),
                "y": ["a" for _ in range(200)],
                "z": [i for i in range(200)],
            }
        )

        validation_results = BayesianRegressionDataValidator().validate(
            "x", ["y", "z"], df, 1
        )
        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }

        assert "CheckNUnique" in validations_dict
        assert validations_dict["CheckNUnique"] == CheckLevels.CRITICAL

    def test_validate_not_CheckNUnique(self):
        df = pd.DataFrame(
            {
                "x": rands_array(10, 5),
                "y": ["a" for _ in range(5)],
                "z": ["b" for i in range(5)],
            }
        )

        validation_results = BayesianRegressionDataValidator().validate(
            "x", ["y", "z"], df, 1
        )
        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }

        assert "CheckNUnique" in validations_dict
        assert validations_dict["CheckNUnique"] == CheckLevels.CRITICAL


class TestCausalDataValidator:
    def test_validate(self):
        df = pd.DataFrame(
            {
                "x": rands_array(10, 5),
                "y": rands_array(10, 5),
                "z": rands_array(10, 5),
                "t": rands_array(10, 5),
            }
        )

        validation_results = CausalDataValidator().validate(["x"], ["y"], df, [], [])

        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "IsSufficientDataChecker" in validations_dict
        assert validations_dict["IsSufficientDataChecker"] == CheckLevels.CRITICAL

    def test_validate_nan_treatment(self):
        df = pd.DataFrame(
            {
                "x": rands_array(100, 5),
                "y": rands_array(100, 5),
                "z": rands_array(100, 5),
                "t": rands_array(100, 5),
            }
        )
        df["x"] = np.nan

        validation_results = CausalDataValidator().validate(["x"], ["y"], df, [], [])

        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "IsSufficientDataChecker" in validations_dict
        assert validations_dict["IsSufficientDataChecker"] == CheckLevels.CRITICAL

    def test_validate_nan_outcome(self):
        df = pd.DataFrame(
            {
                "x": rands_array(100, 5),
                "y": rands_array(100, 5),
                "z": rands_array(100, 5),
                "t": rands_array(100, 5),
            }
        )
        df["y"] = np.nan

        validation_results = CausalDataValidator().validate(["x"], ["y"], df, [], [])

        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "IsSufficientDataChecker" in validations_dict
        assert validations_dict["IsSufficientDataChecker"] == CheckLevels.CRITICAL


class TestRegressionDataValidator:
    def test_validate_columnexistchecker_feature(self):
        df = pd.DataFrame(
            {
                "x": rands_array(10, 5),
                "y": rands_array(10, 5),
                "z": rands_array(10, 5),
                "t": rands_array(10, 5),
            }
        )

        validation_results = RegressionDataValidator().validate("x", ["a"], df, [], [])

        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "ColumnExistsChecker" in validations_dict
        assert validations_dict["ColumnExistsChecker"] == CheckLevels.CRITICAL

    def test_validate_columnexistchecker_target(self):
        df = pd.DataFrame(
            {
                "x": rands_array(10, 5),
                "y": rands_array(10, 5),
                "z": rands_array(10, 5),
                "t": rands_array(10, 5),
            }
        )

        validation_results = RegressionDataValidator().validate("a", ["y"], df, [], [])

        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "ColumnExistsChecker" in validations_dict
        assert validations_dict["ColumnExistsChecker"] == CheckLevels.CRITICAL


class TestClassificationDataValidator:
    def test_validate(self):
        df = pd.DataFrame(
            {
                "x": rands_array(10, 5),
                "y": rands_array(10, 5),
                "z": rands_array(10, 5),
                "t": rands_array(10, 5),
            }
        )

        validation_results = ClassificationDataValidator().validate(
            "a", ["y"], [], [], df, "medium_quality_faster_train"
        )

        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "ColumnExistsChecker" in validations_dict
        assert validations_dict["ColumnExistsChecker"] == CheckLevels.CRITICAL
