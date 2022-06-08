from typing import Any, List, Optional, Union

import pandas as pd
from actableai.causal import has_categorical_column, prepare_sanitize_data
from actableai.data_validation.base import (
    CAUSAL_INFERENCE_CATEGORICAL_MINIMUM_TREATMENT,
    CLASSIFICATION_ANALYTIC,
    CORRELATION_MINIMUM_NUMBER_OF_SAMPLE,
    EXPLAIN_SAMPLES_UNIQUE_CATEGORICAL_LIMIT,
    MINIMUM_NUMBER_OF_SAMPLE,
    POLYNOMIAL_INFLATE_COLUMN_LIMIT,
    REGRESSION_ANALYTIC,
    UNIQUE_CATEGORY_THRESHOLD,
    CheckLevels,
    CheckResult,
)
from actableai.data_validation.checkers import (
    CategoricalSameValuesChecker,
    CategoryChecker,
    CheckColumnInflateLimit,
    CheckNUnique,
    ColumnsExistChecker,
    ColumnsInList,
    ColumnsNotInList,
    CorrectAnalyticChecker,
    DoNotContainDatetimeChecker,
    DoNotContainEmptyColumnsChecker,
    DoNotContainMixedChecker,
    DoNotContainTextChecker,
    InsufficientCategoricalRows,
    IsCategoricalChecker,
    IsCategoricalOrNumericalChecker,
    IsDatetimeChecker,
    IsNumericalChecker,
    IsSufficientClassSampleChecker,
    IsSufficientClassSampleForCrossValidationChecker,
    IsSufficientDataChecker,
    IsSufficientNumberOfClassChecker,
    IsSufficientValidationSampleChecker,
    IsValidFrequencyChecker,
    IsValidNumberOfClusterChecker,
    IsValidPredictionLengthChecker,
    IsValidTypeNumberOfClusterChecker,
    MaxTrainSamplesChecker,
    PositiveOutcomeValueThreshold,
    RegressionEvalMetricChecker,
    SameTypeChecker,
    StratifiedKFoldChecker,
    UniqueDateTimeChecker,
)
from actableai.utils import get_type_special_no_ag


class RegressionDataValidator:
    def validate(
        self,
        target,
        features,
        df,
        debiasing_features,
        debiased_features,
        eval_metric="r2",
        prediction_quantile_low=None,
        prediction_quantile_high=None,
        presets="medium_quality_faster_train",
        explain_samples=False,
        drop_duplicates=True,
    ):
        validation_results = [
            RegressionEvalMetricChecker(level=CheckLevels.CRITICAL).check(eval_metric),
            ColumnsExistChecker(level=CheckLevels.CRITICAL).check(
                df, features + [target]
            ),
        ]

        if (
            len(
                [
                    x
                    for x in validation_results
                    if x is not None and x.level == CheckLevels.CRITICAL
                ]
            )
            > 0
        ):
            return validation_results

        if drop_duplicates:
            df = df.drop_duplicates(subset=features + [target])

        validation_results += [
            DoNotContainEmptyColumnsChecker(level=CheckLevels.WARNING).check(
                df, features
            ),
            DoNotContainEmptyColumnsChecker(level=CheckLevels.CRITICAL).check(
                df, [target]
            ),
            IsSufficientDataChecker(level=CheckLevels.CRITICAL).check(
                df, n_sample=MINIMUM_NUMBER_OF_SAMPLE
            ),
            DoNotContainMixedChecker(level=CheckLevels.WARNING).check(df, features),
            ColumnsNotInList(level=CheckLevels.CRITICAL).check(
                [target], debiasing_features
            ),
            ColumnsNotInList(level=CheckLevels.CRITICAL).check(
                features, debiasing_features
            ),
            ColumnsNotInList(level=CheckLevels.CRITICAL).check(
                debiased_features, debiasing_features
            ),
            ColumnsInList(level=CheckLevels.CRITICAL).check(
                features, debiased_features
            ),
        ]

        if target in df.columns and not pd.isnull(df[target]).all():
            validation_results += [
                IsNumericalChecker(level=CheckLevels.CRITICAL).check(df[target]),
                CorrectAnalyticChecker(level=CheckLevels.WARNING).check(
                    df[target],
                    problem_type=REGRESSION_ANALYTIC,
                    unique_threshold=UNIQUE_CATEGORY_THRESHOLD,
                ),
            ]

        # Check debiasing
        if len(debiasing_features) > 0 and len(debiased_features) <= 0:
            validation_results.append(
                CheckResult(
                    name="DebiasingChecker",
                    level=CheckLevels.CRITICAL,
                    message="At least one debiasing features must be selected",
                )
            )

        run_debiasing = len(debiasing_features) > 0 and len(debiased_features) > 0
        prediction_intervals = (
            prediction_quantile_low is not None or prediction_quantile_high is not None
        )
        # Check prediction intervals
        if run_debiasing and prediction_intervals:
            validation_results.append(
                CheckResult(
                    name="PredictionIntervalChecker",
                    level=CheckLevels.CRITICAL,
                    message="Debiasing is incompatible with prediction intervals",
                )
            )

        # Check presets
        # Note: best_quality is incompatible with debiasing because it activates bagging in AutoGluonj
        if run_debiasing and presets == "best_quality":
            validation_results.append(
                CheckResult(
                    name="PresetsChecker",
                    level=CheckLevels.CRITICAL,
                    message="Optimize for performance is incompatible with debiasing",
                )
            )

        # Check explain samples (incompatible with debiasing)
        if run_debiasing and explain_samples:
            validation_results.append(
                CheckResult(
                    name="ExplanationChecker",
                    level=CheckLevels.CRITICAL,
                    message="Debiasing is incompatible explanation",
                )
            )

        if run_debiasing:
            validation_results.append(
                DoNotContainTextChecker(level=CheckLevels.CRITICAL).check(
                    df, debiasing_features + debiased_features
                )
            )

        return validation_results


class BayesianRegressionDataValidator:
    def validate(
        self, target: str, features: List[str], df: pd.DataFrame, polynomial_degree: int
    ) -> List:

        validation_results = [
            ColumnsExistChecker(level=CheckLevels.CRITICAL).check(df, [target]),
            DoNotContainEmptyColumnsChecker(level=CheckLevels.WARNING).check(
                df, features
            ),
            DoNotContainEmptyColumnsChecker(level=CheckLevels.CRITICAL).check(
                df, [target]
            ),
            IsSufficientDataChecker(level=CheckLevels.CRITICAL).check(
                df, n_sample=MINIMUM_NUMBER_OF_SAMPLE
            ),
            DoNotContainMixedChecker(level=CheckLevels.WARNING).check(df, features),
            # We do not run for more than n_unique_level unique categorical values
            CheckNUnique(level=CheckLevels.CRITICAL).check(
                df=df,
                n_unique_level=EXPLAIN_SAMPLES_UNIQUE_CATEGORICAL_LIMIT,
                analytics="Bayesian Regression",
            ),
            CheckColumnInflateLimit(level=CheckLevels.CRITICAL).check(
                df, features, polynomial_degree, POLYNOMIAL_INFLATE_COLUMN_LIMIT
            ),
        ]

        if target in df.columns and not pd.isnull(df[target]).all():
            validation_results += [
                IsNumericalChecker(level=CheckLevels.CRITICAL).check(df[target]),
                CorrectAnalyticChecker(level=CheckLevels.WARNING).check(
                    df[target],
                    problem_type=REGRESSION_ANALYTIC,
                    unique_threshold=UNIQUE_CATEGORY_THRESHOLD,
                ),
            ]

        return validation_results


class ClassificationDataValidator:
    def validate(
        self,
        target,
        features,
        debiasing_features,
        debiased_features,
        df,
        presets,
        validation_ratio=None,
        kfolds=None,
        drop_duplicates=True,
        explain_samples=False,
    ):
        validation_results = [
            ColumnsExistChecker(level=CheckLevels.CRITICAL).check(
                df, [target, *features, *debiasing_features, *debiased_features]
            )
        ]

        if (
            len(
                [
                    x
                    for x in validation_results
                    if x is not None and x.level == CheckLevels.CRITICAL
                ]
            )
            > 0
        ):
            return validation_results

        if drop_duplicates:
            df = df.drop_duplicates(subset=features + [target])

        validation_results += [
            DoNotContainEmptyColumnsChecker(level=CheckLevels.WARNING).check(
                df, df.columns
            ),
            IsSufficientDataChecker(level=CheckLevels.CRITICAL).check(
                df, n_sample=MINIMUM_NUMBER_OF_SAMPLE
            ),
            DoNotContainMixedChecker(level=CheckLevels.CRITICAL).check(df, features),
            ColumnsNotInList(level=CheckLevels.CRITICAL).check(
                [target], debiasing_features
            ),
            ColumnsNotInList(level=CheckLevels.CRITICAL).check(
                features, debiasing_features
            ),
            ColumnsNotInList(level=CheckLevels.CRITICAL).check(
                debiased_features, debiasing_features
            ),
            ColumnsInList(level=CheckLevels.CRITICAL).check(
                features, debiased_features
            ),
        ]

        if target in df.columns and not pd.isnull(df[target]).all():
            validation_results += [
                IsCategoricalChecker(level=CheckLevels.CRITICAL).check(df[target]),
                IsSufficientNumberOfClassChecker(level=CheckLevels.CRITICAL).check(
                    df[target]
                ),
                CorrectAnalyticChecker(level=CheckLevels.WARNING).check(
                    df[target],
                    problem_type=CLASSIFICATION_ANALYTIC,
                    unique_threshold=UNIQUE_CATEGORY_THRESHOLD,
                ),
            ]

            if validation_ratio is not None:
                validation_results.append(
                    IsSufficientValidationSampleChecker(
                        level=CheckLevels.CRITICAL
                    ).check(df[target], validation_ratio),
                )

            critical_validation_count = 0
            for result in validation_results:
                if result is not None and result.level == CheckLevels.CRITICAL:
                    critical_validation_count += 1

            if critical_validation_count == 0:
                if validation_ratio is not None:
                    validation_results.append(
                        IsSufficientClassSampleChecker(level=CheckLevels.WARNING).check(
                            df, target, validation_ratio
                        )
                    )
                if kfolds > 1:
                    validation_results.append(
                        IsSufficientClassSampleForCrossValidationChecker(
                            level=CheckLevels.CRITICAL
                        ).check(df, target, kfolds)
                    )

        # Check debiasing
        if len(debiasing_features) > 0 and len(debiased_features) <= 0:
            validation_results.append(
                CheckResult(
                    name="DebiasingChecker",
                    level=CheckLevels.CRITICAL,
                    message="At least one debiasing features must be selected",
                )
            )

        run_debiasing = len(debiasing_features) > 0 and len(debiased_features) > 0
        # Check presets
        # Note: best_quality is incompatible with debiasing because it activates bagging in AutoGluonj
        if run_debiasing and presets == "best_quality":
            validation_results.append(
                CheckResult(
                    name="PresetsChecker",
                    level=CheckLevels.CRITICAL,
                    message="Optimize for performance is incompatible with debiasing",
                )
            )

        # Check explain samples (incompatible with debiasing)
        if run_debiasing and explain_samples:
            validation_results.append(
                CheckResult(
                    name="ExplanationChecker",
                    level=CheckLevels.CRITICAL,
                    message="Debiasing is incompatible explanation",
                )
            )

        if run_debiasing:
            validation_results.append(
                DoNotContainTextChecker(level=CheckLevels.CRITICAL).check(
                    df, debiasing_features + debiased_features
                )
            )

        return validation_results


class TimeSeriesDataValidator:
    def __init__(self):
        pass

    def validate(self, df, date_column, predicted_columns, feature_columns, group_by):
        validation_results = [
            ColumnsExistChecker(level=CheckLevels.CRITICAL).check(
                df, [date_column] + predicted_columns + feature_columns + group_by
            ),
            DoNotContainEmptyColumnsChecker(level=CheckLevels.CRITICAL).check(
                df, [date_column] + predicted_columns + feature_columns + group_by
            ),
        ]

        group_df_dict = {}
        if len(group_by) > 0:
            for group, grouped_df in df.groupby(group_by):
                group_df_dict[group] = grouped_df
        else:
            group_df_dict["data"] = df

        for group in group_df_dict.keys():
            df_group = group_df_dict[group].reset_index(drop=True)

            validation_results += [
                IsSufficientDataChecker(level=CheckLevels.CRITICAL).check(
                    df_group, n_sample=MINIMUM_NUMBER_OF_SAMPLE
                ),
                DoNotContainMixedChecker(level=CheckLevels.CRITICAL).check(
                    df_group, predicted_columns
                ),
                CategoryChecker(level=CheckLevels.CRITICAL).check(
                    df_group, predicted_columns
                ),
                DoNotContainTextChecker(level=CheckLevels.CRITICAL).check(
                    df_group, feature_columns
                ),
            ]

            if (date_column in df_group.columns) and not pd.isnull(
                df_group[date_column]
            ).all():
                validation_results += [
                    IsDatetimeChecker(level=CheckLevels.CRITICAL).check(
                        df_group[date_column]
                    ),
                    UniqueDateTimeChecker(level=CheckLevels.CRITICAL).check(
                        df_group[date_column]
                    ),
                    IsValidFrequencyChecker(level=CheckLevels.CRITICAL).check(
                        df_group[date_column]
                    ),
                ]

        return validation_results


class TimeSeriesPredictionDataValidator:
    def validate(
        self,
        group_df_train_dict,
        group_df_valid_dict,
        group_df_predict_dict,
        freq_dict,
        feature_columns,
        predicted_columns,
        prediction_length,
    ):
        validation_results = []
        len_predict = None
        invalid_groups_pred_len = []

        freq = None
        invalid_groups_freq = []
        for group, df_train in group_df_train_dict.items():
            df_valid = group_df_valid_dict.get(group)

            if freq is None:
                freq = freq_dict[group]
            elif freq_dict[group] != freq:
                invalid_groups_freq.append(group)

            df_predict = group_df_predict_dict.get(group)

            if df_predict is not None:
                df_predict_cut = df_predict.loc[~df_predict.index.isin(df_valid.index)]

                if len_predict is None:
                    len_predict = len(df_predict_cut)
                elif len_predict != len(df_predict_cut):
                    invalid_groups_pred_len.append(group)

            if prediction_length is not None:
                validation_results.append(
                    IsValidPredictionLengthChecker(level=CheckLevels.CRITICAL).check(
                        df_train, prediction_length=prediction_length
                    )
                )

            # df_predict cannot be None if there are features
            if len(feature_columns) > 0 and (
                df_predict is None or len(df_predict_cut) == 0
            ):
                validation_results.append(
                    CheckResult(
                        name="PredictionFeaturesChecker",
                        level=CheckLevels.CRITICAL,
                        message="Features for the prediction must be provided",
                    )
                )

            if df_predict is not None and len(feature_columns) > 0:
                if df_predict_cut[predicted_columns].notna().any(axis=None):
                    validation_results.append(
                        CheckResult(
                            name="PredictionLengthChecker",
                            level=CheckLevels.CRITICAL,
                            message="When forecasting with future features, all future values of target prediction columns must be empty",
                        )
                    )

        if len(invalid_groups_pred_len) > 0:
            validation_results.append(
                CheckResult(
                    name="PredictionLengthChecker",
                    level=CheckLevels.CRITICAL,
                    message="Prediction length must be the same for all the groups",
                )
            )

        if len(invalid_groups_freq) > 0:
            validation_results.append(
                CheckResult(
                    name="FrequenciesChecker",
                    level=CheckLevels.CRITICAL,
                    messge="Frequencies must be the same for all the groups",  # type: ignore
                )
            )

        return validation_results


class ClusteringDataValidator:
    def __init__(self):
        pass

    def validate(
        self,
        target,
        df,
        n_cluster,
        explain_samples=False,
        max_train_samples: Optional[int] = None,
    ):
        return [
            ColumnsExistChecker(level=CheckLevels.CRITICAL).check(df, target),
            DoNotContainEmptyColumnsChecker(level=CheckLevels.WARNING).check(
                df, target
            ),
            IsSufficientDataChecker(level=CheckLevels.CRITICAL).check(
                df, n_sample=MINIMUM_NUMBER_OF_SAMPLE
            ),
            DoNotContainMixedChecker(level=CheckLevels.CRITICAL).check(df, target),
            IsValidNumberOfClusterChecker(level=CheckLevels.CRITICAL).check(
                df, n_cluster=n_cluster
            ),
            IsValidTypeNumberOfClusterChecker(level=CheckLevels.CRITICAL).check(
                n_cluster
            ),
            DoNotContainDatetimeChecker(level=CheckLevels.CRITICAL).check(df[target]),
            DoNotContainTextChecker(level=CheckLevels.CRITICAL).check(df, target),
            MaxTrainSamplesChecker(level=CheckLevels.CRITICAL).check(
                n_cluster=n_cluster, max_samples=max_train_samples
            ),
        ]


class CausalDataValidator:
    def __init__(self):
        pass

    def validate(
        self,
        treatments: List[str],
        outcomes: List[str],
        df: pd.DataFrame,
        effect_modifiers: List[str],
        common_causes: List[str],
        positive_outcome_value: Optional[Any],
    ) -> List[Union[CheckResult, None]]:
        columns = effect_modifiers + common_causes
        validation_results = [
            ColumnsExistChecker(level=CheckLevels.CRITICAL).check(df, treatments),
            ColumnsExistChecker(level=CheckLevels.CRITICAL).check(df, outcomes),
            ColumnsExistChecker(level=CheckLevels.CRITICAL).check(df, columns),
        ]
        if len([x for x in validation_results if x is not None]) > 0:
            return validation_results
        # Columns are sane now we treat
        df = prepare_sanitize_data(
            df, treatments, outcomes, effect_modifiers, common_causes
        )
        validation_results += [
            DoNotContainEmptyColumnsChecker(level=CheckLevels.WARNING).check(
                df, columns
            ),
            DoNotContainEmptyColumnsChecker(level=CheckLevels.CRITICAL).check(
                df, treatments
            ),
            DoNotContainEmptyColumnsChecker(level=CheckLevels.CRITICAL).check(
                df, outcomes
            ),
            DoNotContainMixedChecker(level=CheckLevels.WARNING).check(df, columns),
            IsSufficientDataChecker(level=CheckLevels.CRITICAL).check(
                df, n_sample=MINIMUM_NUMBER_OF_SAMPLE
            ),
        ]
        for t in set(treatments):
            validation_results.append(
                DoNotContainMixedChecker(level=CheckLevels.CRITICAL).check(df, [t])
                if t in df.columns and not pd.isnull(df[t]).all()
                else None
            )
        for y in set(outcomes):
            validation_results.append(
                DoNotContainMixedChecker(level=CheckLevels.CRITICAL).check(df, [y])
                if y in df.columns and not pd.isnull(df[y]).all()
                else None
            )

        if CheckLevels.CRITICAL not in [
            check.level for check in validation_results if check is not None
        ]:
            for treatment in treatments:
                if has_categorical_column(df, [treatment]):
                    validation_results.append(
                        InsufficientCategoricalRows(level=CheckLevels.CRITICAL).check(
                            df,
                            treatment=treatment,
                            n_rows=CAUSAL_INFERENCE_CATEGORICAL_MINIMUM_TREATMENT,
                        )
                    )

        if positive_outcome_value is not None:
            validation_results.append(
                PositiveOutcomeValueThreshold(level=CheckLevels.CRITICAL).check(
                    df,
                    outcomes,
                    positive_outcome_value,
                )
            )

        return validation_results


class CorrelationDataValidator:
    def __init__(self):
        pass

    def validate(self, df, target):
        return [
            IsSufficientDataChecker(level=CheckLevels.CRITICAL).check(
                df, CORRELATION_MINIMUM_NUMBER_OF_SAMPLE
            ),
            DoNotContainEmptyColumnsChecker(level=CheckLevels.WARNING).check(
                df, df.columns
            ),
            DoNotContainEmptyColumnsChecker(level=CheckLevels.CRITICAL).check(
                df, target
            ),
        ]


class DataImputationDataValidator:
    def __init__(self):
        pass

    def validate(self, df):
        return [
            IsSufficientDataChecker(level=CheckLevels.WARNING).check(
                df, n_sample=MINIMUM_NUMBER_OF_SAMPLE
            ),
        ]


class InterventionDataValidator:
    def __init__(self):
        pass

    def validate(
        self,
        df,
        target: str,
        current_intervention_column: str,
        new_intervention_column: str,
        common_causes: List[str],
        causal_cv,
    ):
        validations = [
            ColumnsExistChecker(level=CheckLevels.CRITICAL).check(
                df,
                [current_intervention_column, new_intervention_column, target]
                + common_causes,
            )
        ]
        if len([x for x in validations if x is not None]) > 0:
            return validations
        # Columns are sane now we treat
        validations += [
            IsNumericalChecker(level=CheckLevels.CRITICAL).check(df[target]),
            IsCategoricalOrNumericalChecker(level=CheckLevels.CRITICAL).check(
                df, [current_intervention_column, new_intervention_column]
            ),
            SameTypeChecker(level=CheckLevels.CRITICAL).check(
                df, [current_intervention_column, new_intervention_column]
            ),
        ]
        if get_type_special_no_ag(df[current_intervention_column]) == "category":
            validations.append(
                CategoricalSameValuesChecker(level=CheckLevels.CRITICAL).check(
                    df, current_intervention_column, new_intervention_column
                )
            )
            validations.append(
                StratifiedKFoldChecker(level=CheckLevels.CRITICAL).check(
                    df, current_intervention_column, causal_cv
                )
            )

        return validations
