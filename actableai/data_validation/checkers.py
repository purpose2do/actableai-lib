from collections import Counter
from sklearn.preprocessing import PolynomialFeatures
from typing import Any, List, Optional, Union, Type
import pandas as pd
from pandas.api.types import is_numeric_dtype
from actableai.classification.utils import split_validation_by_datetime
from actableai.clustering.models import BaseClusteringModel

from actableai.data_imputation.error_detector.rule_parser import RulesBuilder
from actableai.data_validation.base import (
    CLASSIFICATION_MINIMUM_NUMBER_OF_CLASS_SAMPLE,
    IChecker,
    CheckResult,
    CheckLevels,
)
from actableai.utils import get_type_special_no_ag


class IsNumericalChecker(IChecker):
    def __init__(self, level, name="IsNumericalChecker"):
        self.name = name
        self.level = level

    def check(self, series: pd.Series) -> Optional[CheckResult]:
        """Check if the series is numerical.

        Args:
            series: Series to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        from actableai.utils import get_type_special

        data_type = get_type_special(series)
        if data_type not in ["numeric", "integer"]:
            if data_type == "integer":
                self.level = CheckLevels.WARNING
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"Expected target '{series.name}' to be a numerical column, found {data_type} instead",
            )


class IsCategoricalChecker(IChecker):
    def __init__(self, level, name="IsCategoricalChecker"):
        self.name = name
        self.level = level

    def check(self, df) -> Optional[CheckResult]:
        """Check if the dataframe is categorical.

        Args:
            df: Dataframe to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        from actableai.utils import get_type_special

        data_type = get_type_special(df)  # type: ignore
        if data_type not in ["category", "integer", "boolean"]:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"Expected target '{df.name}' to be a categorical column, found {data_type} instead",
            )


class DoNotContainTextChecker(IChecker):
    def __init__(self, level, name="DoNotContainTextChecker"):
        self.name = name
        self.level = level

    def check(self, df, columns) -> Optional[CheckResult]:
        """Check if the dataframe contains text.

        Args:
            df: Dataframe to check.
            columns: Columns to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        from actableai.utils import get_type_special

        text_columns = []
        for column in columns:
            data_type = get_type_special(df[column])
            if data_type == "text":
                text_columns.append(column)

        if len(text_columns) > 0:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"Columns {', '.join(text_columns)} contain text data type",
            )


class DoNotContainMixedChecker(IChecker):
    def __init__(self, level, name="DoNotContainMixedChecker"):
        self.name = name
        self.level = level

    def check(self, df, columns) -> Optional[CheckResult]:
        """Check if the dataframe contains mixed data types.

        Args:
            df: Dataframe to check.
            columns: Columns to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        mixed_columns = []
        from actableai.utils import get_type_special

        for col in columns:
            if col not in df.columns:
                continue
            data = df[col]
            data_type = get_type_special(data)
            if "mixed" in data_type:
                mixed_columns.append(col)

        if len(mixed_columns) > 0:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"Columns {', '.join(mixed_columns)} contain mixed data type",
            )


class IsDatetimeChecker(IChecker):
    def __init__(self, level, name="IsDatetimeChecker"):
        self.name = name
        self.level = level

    def check(self, df) -> Optional[CheckResult]:
        """Check if the dataframe contains datetime.

        Args:
            df: Dataframe to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        from actableai.utils import get_type_special

        data_type = get_type_special(df)  # type: ignore
        if data_type != "datetime":
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"Expected {df.name} to contains datetime \
                    data type, found {data_type} instead",
            )


class IsSufficientDataChecker(IChecker):
    def __init__(self, level, name="IsSufficientDataChecker"):
        self.name = name
        self.level = level

    def check(self, df, n_sample) -> Optional[CheckResult]:
        """Check if the dataframe contains enough data.

        Args:
            df: Dataframe to check.
            n_sample: Number of samples to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        if len(df) < n_sample:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"The number of data sample is insufficient.\
                    The dataset should have at least {n_sample} samples",
            )


class IsValidTypeNumberOfClusterChecker(IChecker):
    def __init__(self, level, name="IsValidTypeNumberOfClusterChecker"):
        self.name = name
        self.level = level

    def check(self, n_cluster) -> Optional[CheckResult]:
        """Check if the number of cluster is valid.

        Args:
            n_cluster: Number of cluster to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        if type(n_cluster) != int and n_cluster != "auto":
            return CheckResult(
                name=self.name,
                level=self.level,
                message='Number of clusters must be an integer or "auto"',
            )


class IsSufficientClassSampleChecker(IChecker):
    def __init__(self, level, name="IsSufficientClassSampleChecker"):
        self.name = name
        self.level = level

    def check(
        self, df, target, validation_ratio, problem_type="classification"
    ) -> Optional[CheckResult]:
        """Check if each category has enough data.

        Args:
            df: Dataframe to check.
            target: Target column to check.
            validation_ratio (_type_): _description_
            problem_type (str, optional): _description_. Defaults to 'classification'.

        Returns:
            Optional[CheckResult]: _description_
        """
        from actableai.utils import get_type_special
        from sklearn.model_selection import train_test_split
        from autogluon.tabular import TabularPredictor

        col_type = get_type_special(df[target])
        if col_type not in ["category", "integer"]:
            return

        df_for_train = df.groupby(target).filter(
            lambda x: len(x) >= CLASSIFICATION_MINIMUM_NUMBER_OF_CLASS_SAMPLE
        )
        df_for_train = df_for_train[pd.notnull(df_for_train[target])]
        df_for_train = df_for_train.dropna(axis=1, how="all")
        if len(df_for_train) * validation_ratio < df_for_train[target].nunique():
            return CheckResult(
                name=self.name,
                level=CheckLevels.CRITICAL,
                message=f"The number of data sample in validation set\
                {len(df_for_train) * validation_ratio} is insufficient\
                compared to the number of unique values in the target prediction\
                column {df_for_train[target].nunique()}.\
                Please increase the validation ratio or increase the number of examples.",
            )
        train_df, _ = train_test_split(
            df_for_train, test_size=validation_ratio, stratify=df_for_train[target]
        )
        predictor = TabularPredictor(label=target, problem_type=problem_type)
        (
            min_class_sample_threshold,
            _,
            _,
        ) = predictor._learner.adjust_threshold_if_necessary(  # type: ignore
            train_df[target], threshold=10, holdout_frac=0.1, num_bag_folds=0
        )
        valid_df = df.groupby(target).filter(
            lambda x: len(x) < min_class_sample_threshold
        )
        rare_classes = (
            list(valid_df[target].unique().astype(str)) if len(valid_df) > 0 else []
        )
        if len(rare_classes) > 0:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"Rare class(es) ({', '.join(rare_classes)}) \
                    have insufficient numbers of samples and will be removed.\
                    Consider adding more data or lower validation ratio",
            )


class IsSufficientNumberOfClassChecker(IChecker):
    def __init__(self, level, name="IsSufficientNumberOfClassChecker"):
        self.name = name
        self.level = level

    def check(self, target_df) -> Optional[CheckResult]:
        """Check if the number of classes is sufficient.

        Args:
            target_df: Target column to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        n_classes = target_df.nunique()
        if n_classes < 2:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"Minimum number of classes is 2, found {n_classes}",
            )


class IsValidNumberOfClusterChecker(IChecker):
    def __init__(self, level, name="IsValidNumberOfClusterChecker"):
        self.name = name
        self.level = level

    def check(self, df, n_cluster) -> Optional[CheckResult]:
        """Check if the number of cluster is valid against the number of rows.

        Args:
            df: Dataframe to check.
            n_cluster: Number of cluster to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        if type(n_cluster) == int:
            n_sample = len(df)
            if len(df) < n_cluster:
                return CheckResult(
                    name=self.name,
                    level=self.level,
                    message=f"The number of data sample ({n_sample}) should be >= \
                        the number of cluster ({n_cluster}). \
                        Either lower the number of cluster or add more data sample",
                )


class IsValidPredictionLengthChecker(IChecker):
    def __init__(self, level, name="IsValidPredictionLengthChecker"):
        self.name = name
        self.level = level

    def check(self, df, prediction_length) -> Optional[CheckResult]:
        """Check if the prediction length is valid.

        Args:
            df: Dataframe to check.
            prediction_length: Prediction length to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        n_sample = len(df)
        if prediction_length > n_sample / 5:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"Prediction length ({prediction_length}) can not be larger \
                    than 1/5 of the input timeseries ({n_sample}).",
            )


class CategoryChecker(IChecker):
    def __init__(self, level, name="CategoryChecker"):
        self.name = name
        self.level = level

    def check(self, df, columns) -> Optional[CheckResult]:
        """Check if the columns are categorical.

        Args:
            df: Dataframe to check.
            columns: Columns to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        from actableai.utils import get_type_special

        check_cols = [x for x in columns if x in df.columns]
        invalid_cols = []
        for col in check_cols:
            data_type = get_type_special(df[col])
            if not df[col].isnull().all() and data_type == "category":
                invalid_cols.append(col)
        if len(invalid_cols) > 0:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"Category features are not supported yet. Please remove categorical column(s) ({', '.join(invalid_cols)}).",
            )


class ColumnsExistChecker(IChecker):
    def __init__(self, level, name="ColumnsExistChecker"):
        self.name = name
        self.level = level

    def check(self, df, columns) -> Optional[CheckResult]:
        """Check if the columns exist.

        Args:
            df: Dataframe to check.
            columns: Columns to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        check_columns = [columns] if isinstance(columns, str) else columns
        invalid_cols = []
        for col in check_columns:
            if col not in df.columns:
                invalid_cols.append(col)

        if len(invalid_cols) > 0:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"Column(s) ({', '.join(invalid_cols)}) is not in the dataset",
            )


class CheckNUnique(IChecker):
    def __init__(self, level: str, name="CheckNUnique"):
        self.name = name
        self.level = level

    def check(
        self, df: pd.DataFrame, n_unique_level: int, analytics: str = "Explanation"
    ) -> Optional[CheckResult]:
        """Check if the number of unique values is less than the threshold.

        Args:
            df: Dataframe to check.
            n_unique_level: Threshold to check.
            analytics: Type of analytics to use. Either 'Explanation' or
                'Bayesian Regression'.

        Returns:
            Optional[CheckResult]: _description_
        """
        n_unique = df.select_dtypes(include=["object"]).nunique()
        if (n_unique >= n_unique_level).any():
            check_unique_column_name = list(
                n_unique[
                    df.select_dtypes(include=["object"]).nunique() >= n_unique_level
                ].index
            )
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"{analytics} currently doesn't support categorical columns with more than {n_unique_level} unique values.\n"
                + f"{check_unique_column_name} column(s) have too many unique values.",
            )


class ColumnsInList(IChecker):
    def __init__(self, level, name="ColumnsInList"):
        self.name = name
        self.level = level

    def check(self, columns_list, columns) -> Optional[CheckResult]:
        """Check if the columns are in the list.

        Args:
            columns_list: List of columns to check.
            columns: Columns to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        invalid_cols = []

        for col in columns:
            if col not in columns_list:
                invalid_cols.append(col)

        if len(invalid_cols) > 0:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"Column(s) ({', '.join(invalid_cols)}) are not in {', '.join(columns_list)}",
            )


class ColumnsNotInList(IChecker):
    def __init__(self, level, name="ColumnsNotInList"):
        self.name = name
        self.level = level

    def check(self, columns_list, columns) -> Optional[CheckResult]:
        """Check if the columns are not in the list.

        Args:
            columns_list: List of columns to check.
            columns: Columns to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        invalid_cols = []

        for col in columns:
            if col in columns_list:
                invalid_cols.append(col)

        if len(invalid_cols) > 0:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"Column(s) ({', '.join(invalid_cols)}) are in {', '.join(columns_list)}",
            )


class DoNotContainEmptyColumnsChecker(IChecker):
    def __init__(self, level, name="DoNotContainEmptyColumnsChecker"):
        self.name = name
        self.level = level

    def check(self, df, columns) -> Optional[CheckResult]:
        """Check if the columns are full of NaN.

        Args:
            df: Dataframe to check.
            columns: Columns to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        check_columns = [columns] if isinstance(columns, str) else columns
        invalid_cols = []
        for col in check_columns:
            if col not in df.columns:
                continue
            if pd.isnull(df[col]).all():
                invalid_cols.append(col)

        if len(invalid_cols) > 0:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"Empty column(s) ({', '.join(invalid_cols)}) detected",
            )


class IsSufficientValidationSampleChecker(IChecker):
    def __init__(self, level, name="IsSufficientValidationSampleChecker"):
        self.name = name
        self.level = level

    def check(self, df, validation_ratio) -> Optional[CheckResult]:
        """Check if the number of validation samples is greater than the threshold.

        Args:
            df: Dataframe to check.
            validation_ratio: Threshold to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        n_valid_samples = round(df.shape[0] * validation_ratio)
        n_classes = df.nunique()
        if n_valid_samples < n_classes and n_valid_samples > 0:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"The number of validation samples = {n_valid_samples} should be greater or equal to the number of classes = {n_classes}\
                    Please add more data or increase validation percentage",
            )


class CorrectAnalyticChecker(IChecker):
    def __init__(self, level, name="CorrectAnalyticChecker"):
        self.name = name
        self.level = level

    def check(self, df, problem_type, unique_threshold) -> Optional[CheckResult]:
        """Check if you are using the correct analytic. (Classification or Regression)

        Args:
            df: Dataframe to check.
            problem_type: Type of problem to check.
            unique_threshold: Threshold to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        from actableai.utils import get_type_special

        data_type = get_type_special(df)
        if data_type == "integer":
            unique_class = df.nunique()
            suggested_analytic = (
                "classification" if unique_class <= unique_threshold else "regression"
            )
            if suggested_analytic != problem_type:
                return CheckResult(
                    name=self.name,
                    level=self.level,
                    message=f"There are {unique_class} unique classes found in target column. You might want to try {suggested_analytic} analytic instead",
                )


class IsSufficientClassSampleForCrossValidationChecker(IChecker):
    def __init__(self, level, name="IsSufficientClassSampleForCrossValidationChecker"):
        self.name = name
        self.level = level

    def check(self, df, target, kfolds) -> Optional[CheckResult]:
        """Check if the number of validation samples is enough for cross validation.

        Args:
            df: Dataframe to check.
            target: Target column to check.
            kfolds: Number of folds to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        from actableai.utils import get_type_special

        col_type = get_type_special(df[target])
        if col_type not in ["category", "integer"]:
            return

        df_for_train = df.groupby(target).filter(lambda x: len(x) <= kfolds)
        rare_classes = (
            list(df_for_train[target].unique().astype(str))
            if len(df_for_train) > 0
            else []
        )
        if len(rare_classes) > 0:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"Rare class(es) ({', '.join(rare_classes)}) \
                    have insufficient numbers of samples for {kfolds} folds cross validation.\
                    Consider adding more data or lower the number of folds",
            )


class IsValidFrequencyChecker(IChecker):
    def __init__(self, level, name="IsValidFrequencyChecker"):
        self.name = name
        self.level = level

    def check(self, df) -> Optional[CheckResult]:
        """Check if the frequency is valid.

        Args:
            df: Dataframe to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        from actableai.timeseries.utils import find_freq, handle_datetime_column

        try:
            pd_date, _ = handle_datetime_column(df)  # type: ignore
            pd_date.sort_index(inplace=True)
            freq = find_freq(pd_date)
        except Exception:
            freq = None
        if freq is None:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"Datetime column {df.name} has invalid frequency.",
            )


class UniqueDateTimeChecker(IChecker):
    def __init__(self, level, name="IsValidFrequencyChecker"):
        self.name = name
        self.level = level

    def check(self, dt_series) -> Optional[CheckResult]:
        """Check if there is duplicate date time.

        Args:
            dt_series: Series to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        counts = Counter(dt_series)
        dups = dict(filter(lambda item: item[1] > 1, counts.items()))
        if len(dups) > 0:
            return CheckResult(
                name=self.name,
                level=self.level,
                message="Duplicated datetime values:\n"
                + "\n".join([str(dt) for dt in dups.keys()]),
            )


class DoNotContainDatetimeChecker(IChecker):
    def __init__(self, level, name="DoNotContainDatetimeChecker"):
        self.name = name
        self.level = level

    def check(self, df) -> Optional[CheckResult]:
        """Check if the dataframe contains datetime column.

        Args:
            df: Dataframe to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        from actableai.utils import get_type_special

        datetime_columns = []
        for column in df.columns:
            data_type = get_type_special(df[column])
            if data_type == "datetime":
                datetime_columns.append(column)

        if len(datetime_columns) > 0:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"Datetime columns ({', '.join(datetime_columns)}) are not supported.",
            )


class RuleDoNotContainDatetimeChecker(IChecker):
    def __init__(self, level, name="RuleDoNotContainDatetimeChecker"):
        self.name = name
        self.level = level

    def check(self, df, rules) -> Optional[CheckResult]:
        from actableai.utils import get_type_special

        datetime_columns = []
        for column in df.columns:
            data_type = get_type_special(df[column])
            if data_type == "datetime":
                datetime_columns.append(column)

        column_dtypes = dict(df.dtypes.astype(str))
        custom_rules = RulesBuilder.parse(column_dtypes, rules)  # type: ignore
        invalid_columns = []
        invalid_match_rules = [
            x[0] for x in custom_rules.match_rules if x[0] in datetime_columns
        ]
        invalid_columns.extend(invalid_match_rules)

        for constraint in custom_rules.constraints:
            for condition in constraint.when:
                if condition.col1 in datetime_columns:
                    invalid_columns.append(condition.col1)
                elif condition.col2 in datetime_columns:
                    invalid_columns.append(condition.col1)

            for condition in constraint.then:
                if condition.col1 in datetime_columns:
                    invalid_columns.append(condition.col1)
                elif condition.col2 in datetime_columns:
                    invalid_columns.append(condition.col1)

        invalid_columns = list(set(invalid_columns))

        if len(invalid_columns) > 0:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"Datetime columns ({', '.join(invalid_columns)}) are not supported in rules definition.",
            )


class InsufficientCategoricalRows(IChecker):
    def __init__(self, level, name="InsufficientCategoricalRows"):
        self.name = name
        self.level = level

    def check(self, df, treatment, n_rows) -> Optional[CheckResult]:
        """Check if the number of rows is enough for categorical treatment.

        Args:
            df: Dataframe to check.
            treatment: Treatment to check.
            n_rows: Number of rows to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        if (df[treatment].value_counts() < n_rows).any():
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"Categorical treatment ({treatment}) needs at least {n_rows} rows for each different class",
            )


class CheckColumnInflateLimit(IChecker):
    def __init__(self, level: str, name: str = "CheckColumnInflateLimit"):
        self.name = name
        self.level = level

    def check(
        self,
        df: pd.DataFrame,
        features: List[str],
        polynomial_degree: int,
        n_columns: int,
    ) -> Optional[CheckResult]:
        """Check if the number of columns is not too large for the polynomial degree.

        Args:
            df: Dataframe to check.
            features: Features to check.
            polynomial_degree: Polynomial degree for expansion.
            n_columns: Limit number of columns.

        Returns:
            Optional[CheckResult]: _description_
        """
        num_of_cols_dums = (
            df[features].select_dtypes(include=["object"]).nunique().sum()
        )
        inflation_size = PolynomialFeatures._num_combinations(
            num_of_cols_dums, 1, polynomial_degree, False, True
        )
        if inflation_size > n_columns:
            return CheckResult(
                name=self.name,
                level=self.level,
                message="Dataset after inflation is too large. Please lower the polynomial degree or reduce the number of unique values in categorical columns.",
            )


class RegressionEvalMetricChecker(IChecker):
    def __init__(self, level, name="RegressionEvalMetricChecker"):
        self.name = name
        self.level = level

    def check(self, eval_metric: str, use_quantiles: bool) -> Optional[CheckResult]:
        """Check if the eval metric is valid for regression.

        Args:
            eval_metric: Eval metric to check.
            use_quantiles: True if quantiles will be used (different metric).

        Returns:
            Optional[CheckResult]: Check result.
        """
        possible_metrics = []
        if use_quantiles:
            possible_metrics = ["pinball_loss"]
        else:
            possible_metrics = [
                "root_mean_squared_error",
                "mean_squared_error",
                "mean_absolute_error",
                "median_absolute_error",
                "r2",
            ]

        if eval_metric not in possible_metrics:
            return CheckResult(
                name=self.name, level=self.level, message="Invalid eval_metric"
            )


class TimeSeriesTuningMetricChecker(IChecker):
    def __init__(self, level, name="TimeSeriesTuningMetricChecker"):
        self.name = name
        self.level = level

    def check(self, tuning_metric: str) -> Optional[CheckResult]:
        possible_metrics = [
            "abs_error",
            "abs_target_sum",
            "abs_target_mean",
            "seasonal_error",
            "MASE",
            "MAPE",
            "sMAPE",
            "RMSE",
            "ND",
            "mean_absolute_QuantileLoss",
            "mean_wQuantileLoss",
            "MAE_Coverage",
        ]

        if tuning_metric not in possible_metrics:
            return CheckResult(
                name=self.name, level=self.level, message="Invalid tuning_metric"
            )


class MaxTrainSamplesChecker(IChecker):
    def __init__(self, level, name="MaxTrainSamplesChecker"):
        self.name = name
        self.level = level

    def check(
        self, n_cluster: Union[str, int], max_samples: Optional[int]
    ) -> Optional[CheckResult]:
        """Check if the number of samples is not too large for the model.

        Args:
            df: Dataframe to check.
            max_samples: Maximum number of samples.
        Returns:
            Optional[CheckResult]: Check result.
        """

        if (
            isinstance(n_cluster, int)
            and max_samples is not None
            and n_cluster > max_samples
        ):
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"If n_cluster ({n_cluster}) is specified, it should be less than max_samples ({max_samples})",
            )


class PositiveOutcomeValueThreshold(IChecker):
    def __init__(self, level, name="PositiveOutcomeValueThreshold"):
        self.name = name
        self.level = level

    def check(
        self,
        df: pd.DataFrame,
        outcomes: List[str],
        positive_outcome_value: Optional[str],
    ) -> Optional[CheckResult]:
        """Check if the number of samples is enough for the model.

        Args:
            df: Dataframe to check.
            positive_outcome_value: Positive outcome value to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        if positive_outcome_value is not None:
            df[outcomes[0]] = df[outcomes[0]].astype(str)
            positive_outcome_value = str(positive_outcome_value)
            if (df[outcomes[0]] == positive_outcome_value).sum() < 2:
                return CheckResult(
                    name=self.name,
                    level=self.level,
                    message="There should be at least 2 samples with positive outcome"
                    + f"value ({positive_outcome_value}) in the outcome column ({outcomes[0]})",
                )


class IsCategoricalOrNumericalChecker(IChecker):
    def __init__(self, level, name="IsCategoricalNumericalChecker"):
        self.name = name
        self.level = level

    def check(self, df: pd.DataFrame, features: List[str]) -> Optional[CheckResult]:
        """Check if the types of the features are the same.

        Args:
            df: Dataframe to check.
            features: Features to check.
        """

        bad_features = []
        for feature in features:
            if not (
                get_type_special_no_ag(df[feature])
                in ["integer", "numeric", "category"]
            ):
                bad_features.append(feature)
        if len(bad_features) > 0:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"{', '.join(bad_features)} are neither numerical nor categorical",
            )


class SameTypeChecker(IChecker):
    def __init__(self, level, name="SameTypeChecker"):
        self.name = name
        self.level = level

    def check(self, df: pd.DataFrame, features: List[str]) -> Optional[CheckResult]:
        """Check if the features are categorical or numerical.

        Args:
            df: Dataframe to check.
            features: Features to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        og_type = get_type_special_no_ag(df[features[0]])
        if og_type == "integer":
            og_type = "numeric"
        for feature in features:
            feature_type = get_type_special_no_ag(df[feature])
            if feature_type == "integer":
                feature_type = "numeric"
            if feature_type != og_type:
                return CheckResult(
                    name=self.name,
                    level=self.level,
                    message=f"{', '.join(features)} have incompatible types.",
                )


class CategoricalSameValuesChecker(IChecker):
    def __init__(self, level, name="CategoricalSameValuesChecker"):
        self.name = name
        self.level = level

    def check(
        self,
        df: pd.DataFrame,
        current_intervention_column: str,
        new_intervention_column: str,
    ) -> Optional[CheckResult]:
        """Check if the categorical features have the same unique values.

        Args:
            df: Dataframe to check.
            features: Features to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        og_values = df[current_intervention_column].unique()
        for value in df[new_intervention_column].unique():
            if not (value in og_values):
                return CheckResult(
                    name=self.name,
                    level=self.level,
                    message=f"New Intervention : {new_intervention_column} has unseen values than current intervetion : {current_intervention_column}."
                    + "When intervention is categorical please make sure that the new intervention column has the same values as the current intervention column.",
                )


class StratifiedKFoldChecker(IChecker):
    def __init__(self, level, name="StratifiedKFoldChecker"):
        self.name = name
        self.level = level

    def check(
        self, df: pd.DataFrame, intervention: str, causal_cv: int
    ) -> Optional[CheckResult]:
        """Check if the features can be splitted into stratified folds.

        Args:
            df: Dataframe to check.
            features: Features to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        if df[intervention].value_counts().min() < causal_cv:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"The count of each unique values in the column ({intervention}) must be greater than or equal to kfolds number : {causal_cv}.",
            )


class NoFrequentItemSet(IChecker):
    def __init__(self, level: str, name: str = "NoFrequentItemSet"):
        self.name = name
        self.level = level

    def check(self, frequent_itemset: pd.DataFrame) -> Optional[CheckResult]:
        """Check if the frequent item set is empty.

        Args:
            df: Dataframe to check.
            features: Features to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        if len(frequent_itemset) == 0:
            return CheckResult(
                name=self.name,
                level=self.level,
                message="No frequent item set found. Try to lower the minimum value for frequent itemset.",
            )


class ROCAUCChecker(IChecker):
    def __init__(self, level: str, name: str = "ROCAUCChecker"):
        self.name = name
        self.level = level

    def check(
        self, df: pd.DataFrame, target: str, eval_metric: str = "roc_auc"
    ) -> Optional[CheckResult]:
        """Check if the ROC AUC is usable.

        Args:
            df: Dataframe to check.
            features: Features to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        if eval_metric == "roc_auc" and df[target].nunique() > 2:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"ROC AUC eval metric is only available for binary classification. Your target column has {df[target].nunique()} unique values",
            )


class OnlyOneValueChecker(IChecker):
    def __init__(self, level, name="OnlyOneValueChecker"):
        self.name = name
        self.level = level

    def check(self, df: pd.DataFrame, features: List[str]) -> Optional[CheckResult]:
        """Check that all features have only one value.

        Args:
            df: Dataframe to check.
            features: Features to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        if features is None or len(features) == 0:
            return None
        all_unique = (df[features].nunique() == 1).all()
        if all_unique:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"{', '.join(features)} are columns with only one value."
                + " These columns have no predictive power and are removed."
                + " Nothing can be inferred from these columns.",
            )


class SplitByDatetimeValidationChecker(IChecker):
    def __init__(self, level, name="SplitByDatetimeValidationChecker"):
        self.name = name
        self.level = level

    def check(
        self,
        df: pd.DataFrame,
        target: str,
        datetime_column: str,
        validation_ratio: float,
    ) -> Optional[CheckResult]:
        _, df_val = split_validation_by_datetime(df, datetime_column, validation_ratio)
        if df_val[target].nunique() <= 1:
            return CheckResult(
                name=self.name,
                level=self.level,
                message="The validation dataset contains only one value."
                + " Please try to increase the validation ratio or"
                + " use the random data validation split strategy.",
            )


class PositiveOutcomeForBinaryChecker(IChecker):
    def __init__(self, level, name: str = "PositiveOutcomeForBinaryChecker"):
        self.name = name
        self.level = level

    def check(
        self,
        df: pd.DataFrame,
        outcomes: List[str],
        positive_outcome_value: Optional[Any],
    ) -> Optional[CheckResult]:
        """Check that if the target is binary, the positive outcome value is not None.

        Args:
            df: Dataframe to check.
            features: Features to check.

        Returns:
            Optional[CheckResult]: Check result.
        """
        if len(outcomes) == 1:
            outcome = outcomes[0]
            if positive_outcome_value is None and df[outcome].nunique() == 2:
                return CheckResult(
                    name=self.name,
                    level=self.level,
                    message="If the outcome is binary, the positive outcome value must be specified.",
                )


class IsSufficientSampleCrossValidationChecker(IChecker):
    def __init__(self, level, name: str = "IsSufficientSampleCrossValidationChecker"):
        super().__init__(name)
        self.level = level

    def check(self, df: pd.DataFrame, kfolds: int) -> Optional[CheckResult]:
        """Check that there is more values than the number of kfolds

        Args:
            df: Input dataframe
            kfolds: Number of cross validated folds

        Returns:
            Optional[CheckResult]: Check result
        """
        if kfolds > len(df):
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"The number of kfolds ({kfolds}) must be lower or equal to the number of samples ({len(df)})",
            )


class IsSufficientDataClassificationStratification(IChecker):
    def __init__(self, level, name="IsSufficientDataClassificationStratification"):
        super().__init__(name)
        self.level = level

    def check(
        self,
        df: pd.DataFrame,
        target: str,
        validation_ratio: float,
        drop_duplicates: bool,
        features: List[str],
    ) -> Optional[CheckResult]:
        df_train = df[pd.notnull(df[target])]
        if drop_duplicates:
            df_train = df_train.drop_duplicates(subset=features + [target])
        df_train = df_train.groupby(target).filter(
            lambda x: len(x) >= CLASSIFICATION_MINIMUM_NUMBER_OF_CLASS_SAMPLE
        )
        smallest_value = df_train[target].value_counts().min()
        if smallest_value * validation_ratio < 1:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"The validation ratio ({validation_ratio}) is not high enough to represent the target in validation set."
                + f" Please set it to ({1 / smallest_value}+) or add more data.",
            )


class IsSufficientDataTreatmentStratification(IChecker):
    def __init__(self, level, name="IsSufficientDataTreatmentStratification"):
        super().__init__(name)
        self.level = level

    def check(
        self, df: pd.DataFrame, current_intervention_column: str
    ) -> Optional[CheckResult]:
        if df[current_intervention_column].dtype not in [
            "object",
            "category",
            "boolean",
        ]:
            return
        val_counts = df[current_intervention_column].value_counts()
        if (val_counts <= 1).any(axis=None):
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"The current intervention column ({current_intervention_column}) has too few values for each category to perform stratified sampling."
                + "When the current intervention is categorical, the number of samples in each category must be greater than 1.",
            )


class CausalDiscoveryAlgoChecker(IChecker):
    def __init__(self, level, name="CausalDiscoveryAlgoChecker"):
        super().__init__(name)
        self.level = level

    def check(self, algo: str) -> Optional[CheckResult]:
        if algo not in ["deci", "notears", "direct-lingam", "pc"]:
            return CheckResult(
                name=self.name,
                level=self.level,
                message=f"Invalid algorithm: {algo}",
            )


class IsClusteringModelCompatible(IChecker):
    def __init__(self, level, name="IsClusteringAlgorithmCompatible"):
        super().__init__(name)
        self.level = level

    def check(
        self,
        df: pd.DataFrame,
        clustering_model_class: Type[BaseClusteringModel],
    ) -> Optional[CheckResult]:
        if clustering_model_class.handle_categorical:
            return None

        for column in df.columns:
            if not is_numeric_dtype(df[column]):
                return CheckResult(
                    name=self.name,
                    level=self.level,
                    message=f"The current Clustering Model does not correctly support categorical columns, the result might not make sense.",
                )

        return None
