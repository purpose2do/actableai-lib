import numpy as np
import pandas as pd
from pandas._testing import rands_array

from actableai.data_validation.base import CheckLevels
from actableai.data_validation.params import (
    BayesianRegressionDataValidator,
    CausalDataValidator,
    ClusteringDataValidator,
    CorrelationDataValidator,
    InterventionDataValidator,
    RegressionDataValidator,
    ClassificationDataValidator,
    AssociationRulesDataValidator,
    CausalFeatureSelectionDataValidator,
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

        assert "CheckNUnique" not in validations_dict


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

        validation_results = CausalDataValidator().validate(
            ["x"], ["y"], df, [], [], None, False
        )

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

        validation_results = CausalDataValidator().validate(
            ["x"], ["y"], df, [], [], None, False
        )

        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "IsSufficientDataChecker" in validations_dict
        assert validations_dict["IsSufficientDataChecker"] == CheckLevels.CRITICAL

    def test_validate_only_one_value_checker(self):
        df = pd.DataFrame(
            {
                "x": ["a", "a", "a", "a", "a"],
                "y": ["a", "a", "a", "a", "a"],
                "z": ["a", "a", "a", "a", "a"],
                "t": rands_array(10, 5),
            }
        )

        validation_results = CausalDataValidator().validate(
            treatments=["x"],
            outcomes=["t"],
            df=df,
            effect_modifiers=[],
            common_causes=["y", "z"],
            positive_outcome_value=None,
            drop_unique=True,
        )

        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "OnlyOneValueChecker" in validations_dict
        assert validations_dict["OnlyOneValueChecker"] == CheckLevels.CRITICAL

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

        validation_results = CausalDataValidator().validate(
            ["x"], ["y"], df, [], [], None, False
        )

        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "IsSufficientDataChecker" in validations_dict
        assert validations_dict["IsSufficientDataChecker"] == CheckLevels.CRITICAL

    def test_validate_positive_value_outcome(self):
        df = pd.DataFrame(
            {
                "x": rands_array(100, 5),
                "y": [1, 2, 3, 4, 5],
                "z": rands_array(100, 5),
                "t": rands_array(100, 5),
            }
        )

        validation_results = CausalDataValidator().validate(
            ["x"], ["y"], df, [], [], 1, False
        )

        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "PositiveOutcomeValueThreshold" in validations_dict
        assert validations_dict["PositiveOutcomeValueThreshold"] == CheckLevels.CRITICAL


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
        assert "ColumnsExistChecker" in validations_dict
        assert validations_dict["ColumnsExistChecker"] == CheckLevels.CRITICAL

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
        assert "ColumnsExistChecker" in validations_dict
        assert validations_dict["ColumnsExistChecker"] == CheckLevels.CRITICAL

    def test_validate_only_one_value_checker(self):
        df = pd.DataFrame(
            {
                "x": ["a", "a", "a", "a", "a"],
                "y": ["a", "a", "a", "a", "a"],
                "z": ["a", "a", "a", "a", "a"],
                "t": rands_array(10, 5),
            }
        )

        validation_results = RegressionDataValidator().validate(
            target="t",
            features=["x", "y", "z"],
            df=df,
            debiasing_features=[],
            debiased_features=[],
            eval_metric="r2",
            prediction_quantile_low=None,
            prediction_quantile_high=None,
            presets="medium_quality_faster_train",
            explain_samples=False,
            drop_duplicates=True,
        )

        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "OnlyOneValueChecker" in validations_dict
        assert validations_dict["OnlyOneValueChecker"] == CheckLevels.CRITICAL

    def test_validate_datetime_split_none(self):
        df = pd.DataFrame(
            {
                "x": rands_array(10, 5),
                "y": rands_array(10, 5),
                "z": rands_array(10, 5),
                "t": rands_array(10, 5),
            }
        )

        validation_results = RegressionDataValidator().validate(
            target="t",
            features=["x", "y", "z"],
            df=df,
            debiasing_features=[],
            debiased_features=[],
            eval_metric="r2",
            prediction_quantile_low=None,
            prediction_quantile_high=None,
            presets="medium_quality_faster_train",
            explain_samples=False,
            drop_duplicates=True,
            datetime_column=None,
            split_by_datetime=True,
        )

        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "SplitDateTimeColumnExist" in validations_dict
        assert validations_dict["SplitDateTimeColumnExist"] == CheckLevels.CRITICAL

    def test_validate_datetime_split_cross_validation(self):
        df = pd.DataFrame(
            {
                "x": rands_array(10, 5),
                "y": rands_array(10, 5),
                "z": rands_array(10, 5),
                "t": rands_array(10, 5),
            }
        )

        validation_results = RegressionDataValidator().validate(
            target="t",
            features=["x", "y", "z"],
            df=df,
            debiasing_features=[],
            debiased_features=[],
            eval_metric="r2",
            prediction_quantile_low=None,
            prediction_quantile_high=None,
            presets="medium_quality_faster_train",
            explain_samples=False,
            drop_duplicates=True,
            datetime_column="z",
            split_by_datetime=True,
            kfolds=5,
        )

        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "SplitDateTimeCrossValidation" in validations_dict
        assert validations_dict["SplitDateTimeCrossValidation"] == CheckLevels.CRITICAL


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
        assert "ColumnsExistChecker" in validations_dict
        assert validations_dict["ColumnsExistChecker"] == CheckLevels.CRITICAL

    def test_validate_eval_metric_roc_auc(self):
        df = pd.DataFrame(
            {
                "x": rands_array(10, 5),
                "y": rands_array(10, 5),
                "z": rands_array(10, 5),
                "t": rands_array(10, 5),
            }
        )

        validation_results = ClassificationDataValidator().validate(
            target="x",
            features=["y"],
            debiased_features=[],
            debiasing_features=[],
            df=df,
            eval_metric="roc_auc",
            presets="medium_quality_faster_train",
        )

        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "ROCAUCChecker" in validations_dict
        assert validations_dict["ROCAUCChecker"] == CheckLevels.CRITICAL

    def test_validate_only_one_value_checker(self):
        df = pd.DataFrame(
            {
                "x": ["a", "a", "a", "a", "a"],
                "y": ["a", "a", "a", "a", "a"],
                "z": ["a", "a", "a", "a", "a"],
                "t": rands_array(10, 5),
            }
        )

        validation_results = ClassificationDataValidator().validate(
            target="x",
            features=["x", "y", "z"],
            debiased_features=[],
            debiasing_features=[],
            df=df,
            eval_metric="roc_auc",
            presets="medium_quality_faster_train",
        )

        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "OnlyOneValueChecker" in validations_dict
        assert validations_dict["OnlyOneValueChecker"] == CheckLevels.CRITICAL

    def test_validate_datetime_split_none(self):
        df = pd.DataFrame(
            {
                "x": rands_array(10, 5),
                "y": rands_array(10, 5),
                "z": rands_array(10, 5),
                "t": rands_array(10, 5),
            }
        )

        validation_results = ClassificationDataValidator().validate(
            target="t",
            features=["x", "y", "z"],
            df=df,
            debiasing_features=[],
            debiased_features=[],
            eval_metric="r2",
            presets="medium_quality_faster_train",
            explain_samples=False,
            drop_duplicates=True,
            datetime_column=None,
            split_by_datetime=True,
        )

        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "SplitDateTimeColumnExist" in validations_dict
        assert validations_dict["SplitDateTimeColumnExist"] == CheckLevels.CRITICAL

    def test_validate_datetime_split_cross_validation(self):
        df = pd.DataFrame(
            {
                "x": rands_array(10, 5),
                "y": rands_array(10, 5),
                "z": rands_array(10, 5),
                "t": rands_array(10, 5),
            }
        )

        validation_results = RegressionDataValidator().validate(
            target="t",
            features=["x", "y", "z"],
            df=df,
            debiasing_features=[],
            debiased_features=[],
            eval_metric="r2",
            prediction_quantile_low=None,
            prediction_quantile_high=None,
            presets="medium_quality_faster_train",
            explain_samples=False,
            drop_duplicates=True,
            datetime_column="z",
            split_by_datetime=True,
            kfolds=5,
        )

        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "SplitDateTimeCrossValidation" in validations_dict
        assert validations_dict["SplitDateTimeCrossValidation"] == CheckLevels.CRITICAL


class TestCorrelationDataValidator:
    def test_validate(self):
        df = pd.DataFrame(
            {
                "a": [1, 2],
                "b": [1, 2],
            }
        )

        validation_results = CorrelationDataValidator().validate(df, target="a")

        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "IsSufficientDataChecker" in validations_dict
        assert validations_dict["IsSufficientDataChecker"] == CheckLevels.CRITICAL


class TestClusteringDataValidator:
    def test_validate(self):
        df = pd.DataFrame(
            {
                "x": rands_array(10, 5),
                "y": rands_array(10, 5),
                "z": rands_array(10, 5),
                "t": rands_array(10, 5),
            }
        )
        validation_results = ClusteringDataValidator().validate(
            target=["x"], df=df, n_cluster=3, explain_samples=False, max_train_samples=2
        )

        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "MaxTrainSamplesChecker" in validations_dict
        assert validations_dict["MaxTrainSamplesChecker"] == CheckLevels.CRITICAL


class TestInterventionDataValidator:
    def test_validate_column_exists(self):
        df = pd.DataFrame(
            {
                "x": rands_array(10, 5),
                "y": rands_array(10, 5),
                "z": rands_array(10, 5),
                "t": rands_array(10, 5),
            }
        )

        validation_results = InterventionDataValidator().validate(
            df=df,
            target="x",
            current_intervention_column="b",
            new_intervention_column="z",
            common_causes=["t"],
            causal_cv=1,
            drop_unique=False,
        )
        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "ColumnsExistChecker" in validations_dict
        assert validations_dict["ColumnsExistChecker"] == CheckLevels.CRITICAL

    def test_validate_column_identical(self):
        df = pd.DataFrame(
            {
                "x": ["a" for _ in range(5)],
                "y": ["a" for _ in range(5)],
                "z": [1 for _ in range(5)],
                "t": ["a" for _ in range(5)],
            }
        )

        validation_results = InterventionDataValidator().validate(
            df=df,
            target="x",
            current_intervention_column="y",
            new_intervention_column="z",
            common_causes=["t"],
            causal_cv=1,
            drop_unique=False,
        )
        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "SameTypeChecker" in validations_dict
        assert validations_dict["SameTypeChecker"] == CheckLevels.CRITICAL

    def test_validate_target_numerical(self):
        df = pd.DataFrame(
            {
                "x": ["a" for _ in range(5)],
                "y": ["a" for _ in range(5)],
                "z": ["a" for _ in range(5)],
                "t": ["a" for _ in range(5)],
            }
        )

        validation_results = InterventionDataValidator().validate(
            df=df,
            target="x",
            current_intervention_column="y",
            new_intervention_column="z",
            common_causes=["t"],
            causal_cv=1,
            drop_unique=False,
        )
        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "IsNumericalChecker" not in validations_dict

    def test_validate_causal_cv(self):
        df = pd.DataFrame(
            {
                "x": ["a", "b", "c", "d", "e"],
                "y": ["a", "b", "c", "d", "e"],
                "z": ["a", "b", "c", "d", "e"],
                "t": ["a", "b", "c", "d", "e"],
            }
        )

        validation_results = InterventionDataValidator().validate(
            df=df,
            target="x",
            current_intervention_column="y",
            new_intervention_column="z",
            common_causes=["t"],
            causal_cv=2,
            drop_unique=False,
        )

        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "StratifiedKFoldChecker" in validations_dict
        assert validations_dict["StratifiedKFoldChecker"] == CheckLevels.CRITICAL

    def test_validate_only_one_value_checker(self):
        df = pd.DataFrame(
            {
                "x": ["a", "a", "a", "a", "a"],
                "y": ["a", "a", "a", "a", "a"],
                "z": ["a", "a", "a", "a", "a"],
                "t": ["a", "a", "a", "a", "a"],
            }
        )

        validation_results = InterventionDataValidator().validate(
            df=df,
            target="x",
            current_intervention_column="y",
            new_intervention_column="z",
            common_causes=["t"],
            causal_cv=1,
            drop_unique=True,
        )

        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "OnlyOneValueChecker" in validations_dict
        assert validations_dict["OnlyOneValueChecker"] == CheckLevels.CRITICAL


class TestAssociationRulesDataValidator:
    def test_validate_column_present(self):
        df = pd.DataFrame(
            {
                "x": rands_array(10, 5),
                "y": rands_array(10, 5),
                "z": rands_array(10, 5),
                "t": rands_array(10, 5),
            }
        )
        validation_results = AssociationRulesDataValidator().validate(
            df=df, group_by=["x", "a"], items="z"
        )
        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "ColumnsExistChecker" in validations_dict
        assert validations_dict["ColumnsExistChecker"] == CheckLevels.CRITICAL


class TestCausalFeatureSelectionDataVlidator:
    def test_validate_categorical_unique_values(self):
        df = pd.DataFrame(
            {
                "x": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
                "w": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )
        validation_results = CausalFeatureSelectionDataValidator().validate(
            "y", ["x", "w", "z"], df
        )

        validations_dict = {
            val.name: val.level for val in validation_results if val is not None
        }
        assert "CheckNUnique" in validations_dict
        assert validations_dict["CheckNUnique"] == CheckLevels.WARNING
