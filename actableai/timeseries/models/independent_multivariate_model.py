from time import time
from typing import List, Optional, Dict, Tuple, Any, Iterator

import mxnet as mx
import pandas as pd

from actableai.exceptions.timeseries import UntrainedModelException
from actableai.timeseries.dataset import AAITimeSeriesDataset
from actableai.timeseries.models.base import AAITimeSeriesBaseModel
from actableai.timeseries.models.params.base import BaseParams
from actableai.timeseries.models.single_model import AAITimeSeriesSingleModel


class AAITimeSeriesIndependentMultivariateModel(AAITimeSeriesBaseModel):
    """Multi-Target Time Series Model, can be used for univariate and multivariate
    forecasting. It will internally use one AAITimeSeriesSingleModel for each
    target, using the other target as features for every model.

    It also keeps a multivariate model internally, and use it if the performances for
    one specific column are better.
    """

    def __init__(self, prediction_length: int):
        """AAITimeSeriesIndependentMultivariateModel Constructor.

        Args:
            prediction_length: Length of the prediction to forecast.
        """
        super().__init__(prediction_length)

        self.predictor_dict = {}
        self.multivariate_predictor = None

    @staticmethod
    def _get_shift_target_columns(target_columns) -> Dict[str, str]:
        """Create look-up table (dictionary) to associate target columns with their
            shifted corresponding columns.

        Returns:
            The dictionary.
        """
        return {
            target_column: f"_{target_column}_shift" for target_column in target_columns
        }

    def _pre_process_data(
        self,
        dataset: AAITimeSeriesDataset,
        keep_future: bool = True,
    ) -> Tuple[AAITimeSeriesDataset, Dict[str, str]]:
        """Pre-process data to add the shifted target as features.

        Args:
            dataset: Dataset containing the time series.
            keep_future: If False the future (shifted) values are trimmed out.

        Returns:
            - New dictionary containing the time series for each group (along with the
                features).
            - Look-up table to link the columns with their shifted columns.
        """
        if len(dataset.target_columns) == 1:
            return dataset, {}

        shift_target_columns_dict = self._get_shift_target_columns(
            dataset.target_columns
        )
        shift_target_columns = list(shift_target_columns_dict.values())
        feat_dynamic_real_new = list(
            set(dataset.feat_dynamic_real + shift_target_columns)
        )

        df_dict_new = {}
        for group, df in dataset.dataframes.items():
            # Create the shifted dataframe
            df_shift = df.copy()
            if keep_future and dataset.has_dynamic_features:
                df_shift = df.iloc[: -self.prediction_length]

            df_shift = df_shift[dataset.target_columns].shift(
                self.prediction_length, freq=dataset.freq
            )
            # Rename columns
            df_shift = df_shift.rename(columns=shift_target_columns_dict)

            if not keep_future:
                df_shift = df_shift.loc[df_shift.index.isin(df.index)]

            # Add new features
            df_dict_new[group] = pd.concat([df, df_shift], axis=1)

            df_dict_new[group] = df_dict_new[group].iloc[self.prediction_length :]

        return (
            AAITimeSeriesDataset(
                dataframes=df_dict_new,
                target_columns=dataset.target_columns,
                freq=dataset.freq,
                prediction_length=dataset.prediction_length,
                feat_dynamic_real=feat_dynamic_real_new,
                feat_dynamic_cat=dataset.feat_dynamic_cat,
                feat_static_real=dataset.feat_static_real,
                feat_static_cat=dataset.feat_static_cat,
                seasonal_periods=dataset.seasonal_periods,
            ),
            shift_target_columns_dict,
        )

    def _iterate_predictors(
        self,
        dataset_shift: AAITimeSeriesDataset,
        shift_target_columns_dict,
    ) -> Iterator[Tuple[str, AAITimeSeriesBaseModel, AAITimeSeriesDataset]]:
        """Iterate over the predictors, this function handles the shifted columns.

        Args:
            dataset_shift: The dataset to use when iterating.
            shift_target_columns_dict: Look-up table to link the columns with their
                shifted columns.

        Returns:
            An iterator containing the following:
            - Name of the target column.
            - The predictor.
            - The dataset ready for this predictor.
        """
        shift_target_columns = list(shift_target_columns_dict.values())

        for target_column in dataset_shift.target_columns:
            shift_target_column = shift_target_columns_dict.get(target_column)

            feat_dynamic_real = list(
                set(dataset_shift.feat_dynamic_real).difference(
                    set(shift_target_columns)
                )
            )
            feat_dynamic_real_new = list(
                set(dataset_shift.feat_dynamic_real).difference({shift_target_column})
            )

            if target_column not in self.predictor_dict or (
                self.predictor_dict[target_column] is None
                and self.multivariate_predictor is None
            ):
                raise UntrainedModelException()

            if self.predictor_dict[target_column] is None:
                yield target_column, None, None
            else:
                target_dataset = AAITimeSeriesDataset(
                    dataset_shift.dataframes,
                    target_columns=target_column,
                    freq=dataset_shift.freq,
                    prediction_length=dataset_shift.prediction_length,
                    feat_dynamic_real=feat_dynamic_real_new,
                    feat_dynamic_cat=dataset_shift.feat_dynamic_cat,
                    feat_static_real=dataset_shift.feat_static_real,
                    feat_static_cat=dataset_shift.feat_static_cat,
                    seasonal_periods=dataset_shift.seasonal_periods,
                )

                yield target_column, self.predictor_dict[target_column], target_dataset

    def fit(
        self,
        dataset: AAITimeSeriesDataset,
        model_params: List[BaseParams],
        *,
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
            dataset: Dataset containing the time series.
            model_params: List of models parameters to run the tuning search on.
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
        univariate_model_params = []
        multivariate_model_params = []
        for model_param in model_params:
            if model_param.is_multivariate_model:
                multivariate_model_params.append(model_param)
            else:
                univariate_model_params.append(model_param)

        df_target_leaderboard_list = []

        # Fit multivariate
        df_multivariate_agg_metrics = None
        multivariate_fit_time = 0
        if len(dataset.target_columns) > 1 and len(multivariate_model_params) > 0:
            self.multivariate_predictor = AAITimeSeriesSingleModel(
                prediction_length=self.prediction_length,
            )

            (
                multivariate_fit_time,
                df_multivariate_leaderboard,
            ) = self.multivariate_predictor.fit(
                dataset=dataset,
                model_params=multivariate_model_params,
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
                fit_full=False,
            )

            df_target_leaderboard_list.append(df_multivariate_leaderboard)

            _, _, df_multivariate_agg_metrics = self.multivariate_predictor.score(
                dataset=dataset
            )

        # Fit univariate
        dataset_shift, shift_target_columns_dict = self._pre_process_data(
            dataset, keep_future=False
        )

        multi_target_fit_time = 0

        for target_column in dataset.target_columns:
            self.predictor_dict[target_column] = AAITimeSeriesSingleModel(
                prediction_length=self.prediction_length,
            )

        keep_multivariate = False

        # Train one model per target
        for target_column, target_predictor, target_dataset in self._iterate_predictors(
            dataset_shift, shift_target_columns_dict
        ):
            target_total_time, df_target_leaderboard = target_predictor.fit(
                dataset=target_dataset,
                model_params=univariate_model_params,
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
                fit_full=False,
            )

            multi_target_fit_time += target_total_time

            df_target_leaderboard_list.append(df_target_leaderboard)

            if self.multivariate_predictor is not None:
                _, _, df_multi_target_agg_metrics = target_predictor.score(
                    dataset=target_dataset
                )

                multi_target_loss = df_multi_target_agg_metrics[
                    df_multi_target_agg_metrics["target"] == target_column
                ][loss].mean(axis=0)
                multivariate_loss = df_multivariate_agg_metrics[
                    df_multivariate_agg_metrics["target"] == target_column
                ][loss].mean(axis=0)

                # If the multivariate model is better, remove the multi-target model
                if multivariate_loss < multi_target_loss:
                    keep_multivariate = True
                    self.predictor_dict[target_column] = None

        if not keep_multivariate:
            self.multivariate_predictor = None

        fit_full_time = time()
        if fit_full:
            self.refit(dataset, mx_ctx=mx_ctx)
        fit_full_time = time() - fit_full_time

        return multivariate_fit_time + multi_target_fit_time + fit_full_time, pd.concat(
            df_target_leaderboard_list, ignore_index=True
        )

    def refit(
        self,
        dataset: AAITimeSeriesDataset,
        mx_ctx: Optional[mx.Context] = mx.cpu(),
    ):
        """Fit previously tuned model.

        Args:
            dataset: Dataset containing the time series.
            mx_ctx: mxnet context, CPU by default.

        Raises:
            UntrainedModelException: If the model has not been trained/tuned before.
        """
        dataset_shift, shift_target_columns_dict = self._pre_process_data(
            dataset, keep_future=False
        )

        if self.multivariate_predictor is not None:
            self.multivariate_predictor.refit(dataset=dataset, mx_ctx=mx_ctx)

        for _, target_predictor, target_dataset in self._iterate_predictors(
            dataset_shift, shift_target_columns_dict
        ):
            # Skip multivariate model to avoid refitting it multiple times
            if target_predictor is None:
                continue

            target_predictor.refit(dataset=target_dataset, mx_ctx=mx_ctx)

    def score(
        self,
        dataset: AAITimeSeriesDataset,
        num_samples: int = 100,
        quantiles: List[float] = [0.05, 0.5, 0.95],
        num_workers: Optional[int] = None,
    ) -> Tuple[
        Dict[Tuple[Any, ...], pd.DataFrame],
        Dict[Tuple[Any, ...], pd.DataFrame],
        pd.DataFrame,
    ]:
        """Evaluate model.

        Args:
            dataset: Dataset containing the time series.
            num_samples: Number of dataset samples to use for evaluation
            quantiles: List of quantiles to use for evaluation.
            num_workers: Maximum number of workers to use, if None no parallelization
                will be done.

        Raises:
            UntrainedModelException: If the model has not been trained/tuned before.

        Returns:
            - Dictionary containing the predicted time series for each group.
            - Dictionary containing the metrics for each target for each group.
            - Dataframe containing the aggregated metrics for each target.
        """
        dataset_shift, shift_target_columns_dict = self._pre_process_data(
            dataset, keep_future=False
        )

        df_predictions_dict = {
            group: pd.DataFrame() for group in dataset_shift.group_list
        }
        df_item_metrics_dict = {
            group: pd.DataFrame() for group in dataset_shift.group_list
        }
        df_agg_metrics = pd.DataFrame()

        df_multivariate_predictions_dict = None
        df_multivariate_item_metrics_dict = None
        df_multivariate_agg_metrics = None
        if self.multivariate_predictor is not None:
            (
                df_multivariate_predictions_dict,
                df_multivariate_item_metrics_dict,
                df_multivariate_agg_metrics,
            ) = self.multivariate_predictor.score(
                dataset=dataset,
                num_samples=num_samples,
                quantiles=quantiles,
                num_workers=num_workers,
            )

        for target_column, target_predictor, target_dataset in self._iterate_predictors(
            dataset_shift, shift_target_columns_dict
        ):
            if target_predictor is None:
                df_target_predictions_dict = {
                    group: df_pred[df_pred["target"] == target_column]
                    for group, df_pred in df_multivariate_predictions_dict.items()
                }
                df_target_item_metrics_dict = {
                    group: df_im[df_im["target"] == target_column]
                    for group, df_im in df_multivariate_item_metrics_dict.items()
                }
                df_target_agg_metrics = df_multivariate_agg_metrics[
                    df_multivariate_agg_metrics["target"] == target_column
                ]
            else:
                (
                    df_target_predictions_dict,
                    df_target_item_metrics_dict,
                    df_target_agg_metrics,
                ) = target_predictor.score(
                    dataset=target_dataset,
                    num_samples=num_samples,
                    quantiles=quantiles,
                    num_workers=num_workers,
                )

            for group in dataset.group_list:
                df_predictions_dict[group] = pd.concat(
                    [df_predictions_dict[group], df_target_predictions_dict[group]],
                    ignore_index=True,
                )
                df_item_metrics_dict[group] = pd.concat(
                    [df_item_metrics_dict[group], df_target_item_metrics_dict[group]],
                    ignore_index=True,
                )
            df_agg_metrics = pd.concat(
                [df_agg_metrics, df_target_agg_metrics], ignore_index=True
            )

        return df_predictions_dict, df_item_metrics_dict, df_agg_metrics

    def predict(
        self,
        dataset: AAITimeSeriesDataset,
        quantiles: List[float] = [0.05, 0.5, 0.95],
    ) -> Dict[Tuple[Any, ...], pd.DataFrame]:
        """Make a prediction using the model.

        Args:
            dataset: Dataset containing the time series.
            quantiles: Quantiles to predict.

        Raises:
            UntrainedModelException: If the model has not been trained/tuned before.

        Returns:
            Dictionary containing the predicted time series for each group.
        """
        dataset_shift, shift_target_columns_dict = self._pre_process_data(
            dataset, keep_future=True
        )

        df_predictions_dict = {
            group: pd.DataFrame() for group in dataset_shift.group_list
        }

        df_multivariate_predictions_dict = None
        if self.multivariate_predictor is not None:
            df_multivariate_predictions_dict = self.multivariate_predictor.predict(
                dataset=dataset, quantiles=quantiles
            )

        for target_column, target_predictor, target_dataset in self._iterate_predictors(
            dataset_shift, shift_target_columns_dict
        ):
            if target_predictor is None:
                df_target_predictions_dict = {
                    group: df_pred[df_pred["target"] == target_column]
                    for group, df_pred in df_multivariate_predictions_dict.items()
                }
            else:
                df_target_predictions_dict = target_predictor.predict(
                    dataset=target_dataset, quantiles=quantiles
                )

            for group in dataset.group_list:
                df_predictions_dict[group] = pd.concat(
                    [df_predictions_dict[group], df_target_predictions_dict[group]],
                    ignore_index=True,
                )

        return df_predictions_dict
