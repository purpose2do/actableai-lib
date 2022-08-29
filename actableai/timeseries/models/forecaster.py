import mxnet as mx
import pandas as pd
from typing import List, Tuple, Any, Dict, Optional

from actableai.exceptions.timeseries import UntrainedModelException
from actableai.timeseries.dataset import AAITimeSeriesDataset
from actableai.timeseries.models.independent_multivariate_model import (
    AAITimeSeriesIndependentMultivariateModel,
)
from actableai.timeseries.models.params.base import BaseParams


class AAITimeSeriesForecaster:
    """Time Series Forecaster Model."""

    def __init__(
        self,
        date_column: str,
        target_columns: List[str],
        prediction_length: int,
        group_by: List[str] = None,
        feature_columns: List[str] = None,
    ):
        """AAITimeSeriesForecaster Constructor.

        Args:
            date_column: Column containing the date/datetime/time component of the time
                series.
            target_columns: List of columns to forecast, if None all the columns will
                be selected.
            prediction_length: Length of the prediction to forecast.
            group_by: List of columns to use to separate different time series/groups.
            feature_columns: List of columns containing extraneous features used to
                forecast.
        """
        self.date_column = date_column
        self.target_columns = target_columns
        self.prediction_length = prediction_length
        self.group_by = group_by
        self.feature_columns = feature_columns

        if self.group_by is None:
            self.group_by = []
        if self.feature_columns is None:
            self.feature_columns = []

        self.model = None

    @staticmethod
    def _split_static_dynamic_features(
        df_unique: pd.DataFrame,
        real_feature_columns: List[str],
        cat_feature_columns: List[str],
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Split features columns into two groups, static and dynamic.

        Args:
            df_unique: DataFrame containing the number of unique values for each column.
            real_feature_columns: List of real feature columns.
            cat_feature_columns: List of categorical feature columns.

        Returns:
            - List of columns containing real static features.
            - List of columns containing categorical static features.
            - List of columns containing real dynamic features.
            - List of columns containing categorical dynamic features.
        """
        feat_static_real = []
        feat_static_cat = []
        feat_dynamic_real = []
        feat_dynamic_cat = []

        # Real columns
        for column in real_feature_columns:
            if df_unique[column] == 1:
                feat_static_real.append(column)
            else:
                feat_dynamic_real.append(column)

        # Categorical columns
        for column in cat_feature_columns:
            if df_unique[column] == 1:
                feat_static_cat.append(column)
            else:
                feat_dynamic_cat.append(column)

        return (
            feat_static_real,
            feat_static_cat,
            feat_dynamic_real,
            feat_dynamic_cat,
        )

    @classmethod
    def pre_process_data(
        cls,
        df: pd.DataFrame,
        date_column: str,
        target_columns: List[str],
        prediction_length: int,
        feature_columns: List[str] = None,
        group_by: Optional[List[str]] = None,
        inplace: bool = True,
    ) -> AAITimeSeriesDataset:
        """Pre-process dataframe, handle datetime, and return a PandasDataset.

        Args:
            df: Input DataFrame
            date_column: Column containing the date/datetime/time component of the time
                series.
            target_columns: List of columns to forecast, if None all the columns will
                be selected.
            prediction_length: Length of the prediction to forecast.
            feature_columns: List of columns containing extraneous features used to
                forecast.
            group_by: List of columns to use to separate different time series/groups.
            inplace: If True this function will modify the original DataFrame.

        Returns:
            - Dictionary containing the time series for each group.
            - Dictionary containing the frequency of each group.
        """
        df_clean = df

        if not inplace:
            df_clean = df_clean.copy()

        return AAITimeSeriesDataset(
            dataframes=df_clean,
            target_columns=target_columns,
            prediction_length=prediction_length,
            group_by=group_by,
            date_column=date_column,
            feature_columns=feature_columns,
        )

    def fit(
        self,
        model_params: List[BaseParams],
        *,
        df: Optional[pd.DataFrame] = None,
        dataset: Optional[AAITimeSeriesDataset] = None,
        mx_ctx: Optional[mx.Context] = mx.cpu(),
        loss: str = "mean_wQuantileLoss",
        trials: int = 1,
        max_concurrent: Optional[int] = 1,
        use_ray: bool = True,
        tune_samples: int = 3,
        sampling_method: str = "random",
        random_state: Optional[int] = None,
        ray_tune_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        fit_full: bool = True,
    ) -> Tuple[float, pd.DataFrame]:
        """Tune and fit the model.

        Args:
            model_params: List of models parameters to run the tuning search on.
            df: Input DataFrame. If None `group_df_dict`, `freq`, and `group_label_dict`
                must be provided.
            dataset: Dataset containing the time series. If None `df` must be provided.
            mx_ctx: mxnet context, CPU by default.
            loss: Loss to minimize when tuning.
            trials: Number of trials for hyperparameter search.
            max_concurrent: Maximum number of concurrent ray task.
            use_ray: If True ray will be used for hyperparameter tuning.
            tune_samples: Number of dataset samples to use when tuning.
            sampling_method: Method used when extracting the samples for the tuning
                ["random", "last"].
            random_state: Random state to use for reproducibility.
            ray_tune_kwargs: Named parameters to pass to ray's `tune` function.
            verbose: Verbose level.
            fit_full: If True the model will be fit after tuning using all the data
                (tuning data).

        Returns:
            - Total time spent for tuning.
            - Leaderboard
        """
        if df is None and dataset is None:
            raise Exception("df or dataset must be provided")
        if df is not None and dataset is not None:
            raise Exception("df or dataset must be provided")

        if dataset is None:
            dataset = self.pre_process_data(
                df=df,
                date_column=self.date_column,
                target_columns=self.target_columns,
                prediction_length=self.prediction_length,
                feature_columns=self.feature_columns,
                group_by=self.group_by,
                inplace=False,
            )

        # Train multi-target model
        self.model = AAITimeSeriesIndependentMultivariateModel(
            prediction_length=self.prediction_length,
        )
        multi_target_fit_time, df_multi_target_leaderboard = self.model.fit(
            dataset=dataset,
            model_params=model_params,
            mx_ctx=mx_ctx,
            loss=loss,
            trials=trials,
            max_concurrent=max_concurrent,
            use_ray=use_ray,
            tune_samples=tune_samples,
            sampling_method=sampling_method,
            random_state=random_state,
            ray_tune_kwargs=ray_tune_kwargs,
            verbose=verbose,
            fit_full=fit_full,
        )

        return multi_target_fit_time, df_multi_target_leaderboard

    def refit(
        self,
        *,
        df: Optional[pd.DataFrame] = None,
        dataset: Optional[AAITimeSeriesDataset] = None,
        mx_ctx: Optional[mx.Context] = mx.cpu(),
    ):
        """Fit previously tuned model.

        Args:
            df: Input DataFrame. If None `group_df_dict` must be provided.
            dataset: Dataset containing the time series. If None `df` must be provided.
            mx_ctx: mxnet context, CPU by default.

        Raises:
            UntrainedModelException: If the model has not been trained/tuned before.
        """
        if self.model is None:
            raise UntrainedModelException()

        if df is None and dataset is None:
            raise Exception("df or dataset must be provided")
        if df is not None and dataset is not None:
            raise Exception("df or dataset must be provided")

        if dataset is None:
            dataset = self.pre_process_data(
                df=df,
                date_column=self.date_column,
                target_columns=self.target_columns,
                prediction_length=self.prediction_length,
                feature_columns=self.feature_columns,
                group_by=self.group_by,
                inplace=False,
            )

        self.model.refit(dataset=dataset, mx_ctx=mx_ctx)

    def score(
        self,
        *,
        df: Optional[pd.DataFrame] = None,
        dataset: Optional[AAITimeSeriesDataset] = None,
        num_samples: int = 100,
        quantiles: List[float] = [0.05, 0.5, 0.95],
        num_workers: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Evaluate model.

        Args:
            df: Input DataFrame. If None `group_df_dict` must be provided.
            dataset: Dataset containing the time series. If None `df` must be provided.
            num_samples: Number of dataset samples to use for evaluation
            quantiles: List of quantiles to use for evaluation.
            num_workers: Maximum number of workers to use, if None no parallelization
                will be done.

        Raises:
            UntrainedModelException: If the model has not been trained/tuned before.

        Returns:
            - Predicted time series.
            - Metrics for each target and groups.
            - Aggregated metrics for each target.
        """
        if self.model is None:
            raise UntrainedModelException()

        if df is None and dataset is None:
            raise Exception("df or dataset must be provided")
        if df is not None and dataset is not None:
            raise Exception("df or dataset must be provided")

        if dataset is None:
            dataset = self.pre_process_data(
                df=df,
                date_column=self.date_column,
                target_columns=self.target_columns,
                prediction_length=self.prediction_length,
                feature_columns=self.feature_columns,
                group_by=self.group_by,
                inplace=False,
            )

        df_predictions_dict, df_item_metrics_dict, df_agg_metrics = self.model.score(
            dataset=dataset,
            num_samples=num_samples,
            quantiles=quantiles,
            num_workers=num_workers,
        )

        # Post process scores and predictions
        df_predictions = pd.DataFrame()
        for group, df_group in df_predictions_dict.items():
            df_group["_group"] = [group] * len(df_group)
            df_predictions = pd.concat([df_predictions, df_group], ignore_index=True)

        df_item_metrics = pd.DataFrame()
        for group, df_group in df_item_metrics_dict.items():
            df_group["_group"] = [group] * len(df_group)
            df_item_metrics = pd.concat([df_item_metrics, df_group], ignore_index=True)

        for group_index, group in enumerate(self.group_by):
            f_group_values = lambda group_values: group_values[group_index]
            df_predictions[group] = df_predictions["_group"].apply(f_group_values)
            df_item_metrics[group] = df_predictions["_group"].apply(f_group_values)

        df_predictions = df_predictions.rename(columns={"date": self.date_column}).drop(
            columns="_group"
        )
        df_item_metrics = df_item_metrics.drop(columns="_group")

        return df_predictions, df_item_metrics, df_agg_metrics

    def predict(
        self,
        *,
        df: Optional[pd.DataFrame] = None,
        dataset: Optional[AAITimeSeriesDataset] = None,
        quantiles: List[float] = [0.05, 0.5, 0.95],
    ) -> pd.DataFrame:
        """Make a prediction using the model.

        Args:
            df: Input DataFrame. If None `group_df_dict` must be provided.
            dataset: Dataset containing the time series. If None `df` must be provided.
            quantiles: Quantiles to predict.

        Raises:
            UntrainedModelException: If the model has not been trained/tuned before.

        Returns:
            Predicted time series.
        """
        if self.model is None:
            raise UntrainedModelException()

        if df is None and dataset is None:
            raise Exception("df or dataset must be provided")
        if df is not None and dataset is not None:
            raise Exception("df or dataset must be provided")

        if dataset is None:
            dataset = self.pre_process_data(
                df=df,
                date_column=self.date_column,
                target_columns=self.target_columns,
                prediction_length=self.prediction_length,
                feature_columns=self.feature_columns,
                group_by=self.group_by,
                inplace=False,
            )

        df_predictions_dict = self.model.predict(dataset=dataset, quantiles=quantiles)

        # Post process predictions
        df_predictions = pd.DataFrame()
        for group, df_group in df_predictions_dict.items():
            df_group["_group"] = [group] * len(df_group)
            df_predictions = pd.concat([df_predictions, df_group], ignore_index=True)

        for group_index, group in enumerate(self.group_by):
            f_group_values = lambda group_values: group_values[group_index]
            df_predictions[group] = df_predictions["_group"].apply(f_group_values)

        df_predictions = df_predictions.rename(columns={"date": self.date_column}).drop(
            columns="_group"
        )

        return df_predictions
