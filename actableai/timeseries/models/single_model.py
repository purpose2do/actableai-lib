import mxnet as mx
import numpy as np
import pandas as pd
import time
import visions
from copy import deepcopy
from functools import partial
from gluonts.mx.distribution import DistributionOutput
from gluonts.mx.distribution.poisson import PoissonOutput
from gluonts.mx.distribution.student_t import StudentTOutput
from hyperopt import hp, fmin, tpe, space_eval, Trials
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from typing import Dict, List, Optional, Tuple, Any, Iterable, Union

from actableai.exceptions.timeseries import UntrainedModelException
from actableai.timeseries.models.base import AAITimeSeriesBaseModel
from actableai.timeseries.models.estimator import AAITimeSeriesEstimator
from actableai.timeseries.models.params.base import BaseParams
from actableai.timeseries.models.predictor import AAITimeSeriesPredictor
from actableai.timeseries.models.evaluator import AAITimeSeriesEvaluator
from actableai.timeseries.utils import (
    dataframe_to_list_dataset,
    forecast_to_dataframe,
)


class AAITimeSeriesSingleModel(AAITimeSeriesBaseModel):
    """Simple Time Series Model,"""

    def __init__(
        self,
        target_columns: List[str],
        prediction_length: int,
        freq: str,
        group_label_dict: Optional[Dict[Tuple[Any, ...], int]] = None,
        real_static_feature_dict: Optional[Dict[Tuple[Any, ...], List[float]]] = None,
        cat_static_feature_dict: Optional[Dict[Tuple[Any, ...], List[Any]]] = None,
        real_dynamic_feature_columns: Optional[List[str]] = None,
        cat_dynamic_feature_columns: Optional[List[str]] = None,
    ):
        """AAITimeSeriesBaseModel Constructor.

        Args:
            target_columns: List of columns to forecast.
            prediction_length: Length of the prediction to forecast.
            freq: Frequency of the time series.
            group_label_dict: Dictionary containing the unique label for each group.
            real_static_feature_dict: Dictionary containing a list of real static
                features for each group.
            cat_static_feature_dict: Dictionary containing a list of categorical static
                features for each group.
            real_dynamic_feature_columns: List of columns containing real dynamic
                features.
            cat_dynamic_feature_columns: List of columns containing categorical dynamic
                features.
        """
        super().__init__(
            target_columns,
            prediction_length,
            freq,
            group_label_dict,
            real_static_feature_dict,
            cat_static_feature_dict,
            real_dynamic_feature_columns,
            cat_dynamic_feature_columns,
        )

        self.predictor = None
        self.best_params = None

        self.model_params_dict = None
        self.distr_output = None

    @staticmethod
    def _create_predictor(
        model_params_dict: Dict[str, BaseParams],
        params: Dict[str, Any],
        data: Iterable[Dict[str, Any]],
        freq_gluon: str,
        distr_output: DistributionOutput,
        prediction_length: int,
        target_dim: int,
        mx_ctx: mx.Context,
    ) -> AAITimeSeriesPredictor:
        """Create and train a predictor.

        Args:
            model_params_dict: Dictionary containing the different model params for
                each model.
            params: Hyperparameter choose by the tuning.
            data: Data to use for training.
            freq_gluon: GluonTS frequency of the time series.
            distr_output: Distribution output to use.
            prediction_length: Length of the prediction that will be forecasted.
            target_dim: Target dimension (number of columns to predict).
            mx_ctx: mxnet context.

        Returns:
            Trained predictor.
        """
        model_params_class = model_params_dict[params["model"]["model_name"]]

        keep_feat_static_real = model_params_class.handle_feat_static_real
        keep_feat_static_cat = model_params_class.handle_feat_static_cat
        keep_feat_dynamic_real = model_params_class.handle_feat_dynamic_real
        keep_feat_dynamic_cat = model_params_class.handle_feat_dynamic_cat

        if model_params_class.has_estimator:
            gluonts_estimator = model_params_class.build_estimator(
                ctx=mx_ctx,
                freq=freq_gluon,
                prediction_length=prediction_length,
                target_dim=target_dim,
                distr_output=distr_output,
                params=params["model"],
            )

            estimator = AAITimeSeriesEstimator(
                gluonts_estimator,
                keep_feat_static_real,
                keep_feat_static_cat,
                keep_feat_dynamic_real,
                keep_feat_dynamic_cat,
            )

            predictor = estimator.train(training_data=data)
        else:
            predictor = model_params_class.build_predictor(
                freq=freq_gluon,
                prediction_length=prediction_length,
                params=params["model"],
            )

        return AAITimeSeriesPredictor(
            predictor,
            keep_feat_static_real,
            keep_feat_static_cat,
            keep_feat_dynamic_real,
            keep_feat_dynamic_cat,
        )

    @classmethod
    def _trainable(
        cls,
        params: Dict[str, Any],
        *,
        model_params_dict: Dict[str, BaseParams],
        train_data_partial: Iterable[Dict[str, Any]],
        tune_data: Iterable[Dict[str, Any]],
        loss: str,
        freq_gluon: str,
        distr_output: DistributionOutput,
        prediction_length: int,
        target_columns: List[str],
        use_ray: bool,
        mx_ctx: mx.Context,
    ) -> Optional[pd.DataFrame]:
        """Create, train, and evaluate a model with specific hyperparameter.

        Args:
            params: Hyperparameter choose by the tuning.
            model_params_dict: Dictionary containing the different model params for
                each model.
            train_data_partial: Data to use for training.
            tune_data: Data to use for tuning.
            loss: Loss to return.
            freq_gluon: GluonTS frequency of the time series.
            distr_output: Distribution output to use.
            prediction_length: Length of the prediction that will be forecasted.
            target_columns: List of columns to forecast.
            use_ray: Whether ray is used for tuning or not.
            mx_ctx: mxnet context.

        Returns:
            If `use_ray` is False return the loss. Else will report the loss to ray
            tune.
        """
        predictor = cls._create_predictor(
            model_params_dict,
            params,
            train_data_partial,
            freq_gluon,
            distr_output,
            prediction_length,
            len(target_columns),
            mx_ctx,
        )

        forecast_it, ts_it = predictor.make_evaluation_predictions(
            tune_data, num_samples=100
        )

        evaluator = AAITimeSeriesEvaluator(
            target_columns=target_columns,
            quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
            num_workers=None,
        )
        _, df_agg_metrics = evaluator(ts_it, forecast_it, num_series=len(tune_data))

        if not use_ray:
            return df_agg_metrics

        tune.report(
            **{
                loss: df_agg_metrics[loss].mean(),
                "df_agg_metrics": df_agg_metrics.to_dict(),
            }
        )

    @classmethod
    def _objective(
        cls,
        params: Dict[str, Any],
        *,
        model_params_dict: Dict[str, BaseParams],
        train_data_partial: Iterable[Dict[str, Any]],
        tune_data: Iterable[Dict[str, Any]],
        loss: str,
        freq_gluon: str,
        distr_output: DistributionOutput,
        prediction_length: int,
        target_columns: List[str],
        use_ray: bool,
        mx_ctx: mx.Context,
    ) -> Dict[str, Any]:
        """Create, train, and evaluate a model with specific hyperparameter. Used by
            hyperopt.

        Args:
            params: Hyperparameter choose by the tuning.
            model_params_dict: Dictionary containing the different model params for
                each model.
            train_data_partial: Data to use for training.
            tune_data: Data to use for tuning.
            loss: Loss to return.
            freq_gluon: GluonTS frequency of the time series.
            distr_output: Distribution output to use.
            target_columns: List of columns to forecast.
            target_dim: Target dimension (number of columns to predict).
            use_ray: Whether ray is used for tuning or not.
            mx_ctx: mxnet context.

        Returns:
            Dictionary containing the loss and the status.
        """
        df_agg_metrics = cls._trainable(
            params,
            model_params_dict=model_params_dict,
            train_data_partial=train_data_partial,
            tune_data=tune_data,
            loss=loss,
            freq_gluon=freq_gluon,
            distr_output=distr_output,
            prediction_length=prediction_length,
            target_columns=target_columns,
            use_ray=use_ray,
            mx_ctx=mx_ctx,
        )

        return {
            "loss": df_agg_metrics[loss].mean(),
            "df_agg_metrics": df_agg_metrics.to_dict(),
            "status": "ok",
        }

    def _generate_train_valid_data(
        self,
        group_df_dict: Dict[Tuple[Any, ...], pd.DataFrame],
        tune_samples: int,
        sampling_method: str = "random",
    ) -> Tuple[
        Iterable[Dict[str, Any]], Iterable[Dict[str, Any]], Iterable[Dict[str, Any]]
    ]:
        """Generate and split train and validation data for tuning.

        Args:
            group_df_dict: Dictionary containing the time series for each group.
            tune_samples: Number of dataset samples to use when tuning.
            sampling_method: Method used when extracting the samples for the tuning
                ["random", "last"].

        Returns:
            - Training ListDataset.
            - Training ListDataset (partial without tuning).
            - Tuning ListDataset.
        """
        from gluonts.dataset.common import ListDataset

        train_data = dataframe_to_list_dataset(
            group_df_dict,
            self.target_columns,
            self.freq_gluon,
            real_static_feature_dict=self.real_static_feature_dict,
            cat_static_feature_dict=self.cat_static_feature_dict,
            real_dynamic_feature_columns=self.real_dynamic_feature_columns,
            cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
            group_label_dict=self.group_label_dict,
            prediction_length=self.prediction_length,
            training=True,
        )

        train_data_partial = dataframe_to_list_dataset(
            group_df_dict,
            self.target_columns,
            self.freq_gluon,
            real_static_feature_dict=self.real_static_feature_dict,
            cat_static_feature_dict=self.cat_static_feature_dict,
            real_dynamic_feature_columns=self.real_dynamic_feature_columns,
            cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
            group_label_dict=self.group_label_dict,
            prediction_length=self.prediction_length,
            slice_df=slice(-self.prediction_length - tune_samples),
            training=True,
        )

        tune_data_list = []
        for i in range(tune_samples):
            slice_function = None
            if sampling_method == "random":
                slice_function = lambda df: slice(
                    np.random.randint(3 * self.prediction_length + 1, df.shape[0] + 1)
                )
            elif sampling_method == "last":
                slice_function = lambda df: slice(df.shape[0] - i)
            else:
                raise Exception("Unkown sampling method")

            tune_data_list.append(
                dataframe_to_list_dataset(
                    group_df_dict,
                    self.target_columns,
                    self.freq_gluon,
                    real_static_feature_dict=self.real_static_feature_dict,
                    cat_static_feature_dict=self.cat_static_feature_dict,
                    real_dynamic_feature_columns=self.real_dynamic_feature_columns,
                    cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
                    group_label_dict=self.group_label_dict,
                    prediction_length=self.prediction_length,
                    slice_df=slice_function,
                    training=True,
                )
            )

        # Merge all samples into the same ListDataset
        tune_data = None
        if len(tune_data_list) > 0:
            list_data = []

            for tune_data_sample in tune_data_list:
                list_data += tune_data_sample.list_data

            tune_data = ListDataset(
                list_data,
                self.freq_gluon,
                one_dim_target=(len(self.target_columns) == 1),
            )

        return train_data, train_data_partial, tune_data

    def fit(
        self,
        group_df_dict: Dict[Tuple[Any, ...], pd.DataFrame],
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
            group_df_dict: Dictionary containing the time series for each group.
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
        self.model_params_dict = {
            model_param_class.model_name: model_param_class
            for model_param_class in model_params
            if model_param_class is not None
        }

        # Split data
        train_data, train_data_partial, tune_data = self._generate_train_valid_data(
            group_df_dict,
            tune_samples,
            sampling_method,
        )

        # Choose distribution output
        first_group_targets = train_data.list_data[0]["target"]
        if (first_group_targets >= 0).all() and first_group_targets in visions.Integer:
            self.distr_output = PoissonOutput()
        else:
            self.distr_output = StudentTOutput()

        # Set up search space
        models_search_space = []
        for model_name, model_param_class in self.model_params_dict.items():
            if model_param_class is not None:
                models_search_space.append(
                    {
                        "model_name": model_name,
                        **model_param_class.tune_config(),
                    }
                )
        search_space = {"model": hp.choice("model", models_search_space)}

        trials_time_total = 0
        trials_result_list = []

        # Tune hyperparameters
        if use_ray:
            algo = HyperOptSearch(
                search_space, metric=loss, mode="min", random_state_seed=random_state
            )
            if max_concurrent is not None:
                algo = ConcurrencyLimiter(algo, max_concurrent=max_concurrent)

            if ray_tune_kwargs is None:
                ray_tune_kwargs = {}

            trainable = tune.with_parameters(
                self._trainable,
                model_params_dict=self.model_params_dict,
                train_data_partial=train_data_partial,
                tune_data=tune_data,
                loss=loss,
                freq_gluon=self.freq_gluon,
                distr_output=self.distr_output,
                prediction_length=self.prediction_length,
                target_columns=self.target_columns,
                use_ray=use_ray,
                mx_ctx=mx_ctx,
            )

            analysis = tune.run(
                trainable,
                search_alg=algo,
                num_samples=trials,
                verbose=verbose,
                **ray_tune_kwargs,
            )

            # Build leaderboard
            for result in analysis.results.values():
                if result is None or loss not in result:
                    continue

                trial_training_time = result.get("time_total_s", 0)
                trials_time_total += trial_training_time

                config = deepcopy(result["config"]["model"])
                model_name = config["model_name"]
                del config["model_name"]

                trials_result_list.append(
                    {
                        "target": None,
                        "model_name": model_name,
                        "model_parameters": str(config),
                        "training_time": trial_training_time,
                        **pd.DataFrame.from_dict(result["df_agg_metrics"])
                        .mean()
                        .to_dict(),
                    }
                )

            start = time.time()

            self.best_params = analysis.get_best_config(metric=loss, mode="min")
        else:
            start = time.time()

            objective = partial(
                self._objective,
                model_params_dict=self.model_params_dict,
                train_data_partial=train_data_partial,
                tune_data=tune_data,
                loss=loss,
                freq_gluon=self.freq_gluon,
                distr_output=self.distr_output,
                prediction_length=self.prediction_length,
                target_columns=self.target_columns,
                use_ray=use_ray,
                mx_ctx=mx_ctx,
            )

            trials_results = Trials()
            best = fmin(
                fn=objective,
                space=search_space,
                algo=tpe.suggest,
                max_evals=trials,
                trials=trials_results,
            )

            # Build leaderboard
            for trial in trials_results.trials:
                vals = {
                    key: value[0]
                    for key, value in trial["misc"]["vals"].items()
                    if len(value) > 0
                }

                model_parameters = space_eval(space=search_space, hp_assignment=vals)[
                    "model"
                ]
                model_name = model_parameters["model_name"]
                del model_parameters["model_name"]

                trials_result_list.append(
                    {
                        "target": None,
                        "model_name": model_name,
                        "model_parameters": str(model_parameters),
                        "training_time": (
                            trial["refresh_time"] - trial["book_time"]
                        ).total_seconds(),
                        **pd.DataFrame.from_dict(trial["result"]["df_agg_metrics"])
                        .mean()
                        .to_dict(),
                    }
                )

            self.best_params = space_eval(space=search_space, hp_assignment=best)

        df_leaderboard = pd.DataFrame(trials_result_list)
        df_leaderboard["target"] = (
            str(self.target_columns)
            if len(self.target_columns) > 1
            else self.target_columns[0]
        )

        if fit_full:
            final_train_data = train_data
        else:
            final_train_data = train_data_partial

        # Create final model using best parameters
        self.predictor = self._create_predictor(
            model_params_dict=self.model_params_dict,
            params=self.best_params,
            data=final_train_data,
            freq_gluon=self.freq_gluon,
            distr_output=self.distr_output,
            prediction_length=self.prediction_length,
            target_dim=len(self.target_columns),
            mx_ctx=mx_ctx,
        )

        return time.time() - start + trials_time_total, df_leaderboard

    def refit(
        self,
        group_df_dict: Dict[Tuple[Any, ...], pd.DataFrame],
        mx_ctx: Optional[mx.Context] = mx.cpu(),
    ):
        """Fit previously tuned model.

        Args:
            group_df_dict: Dictionary containing the time series for each group.
            mx_ctx: mxnet context, CPU by default.

        Raises:
            UntrainedModelException: If the model has not been trained/tuned before.
        """
        if self.predictor is None:
            raise UntrainedModelException()

        train_data = dataframe_to_list_dataset(
            group_df_dict,
            self.target_columns,
            self.freq_gluon,
            real_static_feature_dict=self.real_static_feature_dict,
            cat_static_feature_dict=self.cat_static_feature_dict,
            real_dynamic_feature_columns=self.real_dynamic_feature_columns,
            cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
            group_label_dict=self.group_label_dict,
            prediction_length=self.prediction_length,
            training=True,
        )

        self.predictor = self._create_predictor(
            model_params_dict=self.model_params_dict,
            params=self.best_params,
            data=train_data,
            freq_gluon=self.freq_gluon,
            distr_output=self.distr_output,
            prediction_length=self.prediction_length,
            target_dim=len(self.target_columns),
            mx_ctx=mx_ctx,
        )

    def score(
        self,
        group_df_dict: Dict[Tuple[Any, ...], pd.DataFrame],
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
            group_df_dict: Dictionary containing the time series for each group.
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
        if self.predictor is None:
            raise UntrainedModelException()

        df_predictions_dict = {}

        valid_data = dataframe_to_list_dataset(
            group_df_dict,
            self.target_columns,
            self.freq_gluon,
            real_static_feature_dict=self.real_static_feature_dict,
            cat_static_feature_dict=self.cat_static_feature_dict,
            real_dynamic_feature_columns=self.real_dynamic_feature_columns,
            cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
            group_label_dict=self.group_label_dict,
            prediction_length=self.prediction_length,
            training=True,
        )

        forecast_it, ts_it = self.predictor.make_evaluation_predictions(
            valid_data, num_samples
        )

        ts_list = list(ts_it)
        forecast_list = list(forecast_it)

        # Create custom metric function
        rmse = lambda target, forecast: np.sqrt(np.mean(np.square(target - forecast)))

        # Evaluate
        evaluator = AAITimeSeriesEvaluator(
            target_columns=self.target_columns,
            group_list=list(group_df_dict.keys()),
            quantiles=quantiles,
            num_workers=num_workers,
            custom_eval_fn={"custom_RMSE": [rmse, "mean", "median"]},
        )
        df_item_metrics_dict, df_agg_metrics = evaluator(
            ts_list, forecast_list, num_series=len(valid_data)
        )

        # Add predictions
        for (group, df_group), forecast in zip(group_df_dict.items(), forecast_list):
            df_predictions_dict[group] = forecast_to_dataframe(
                forecast,
                self.target_columns,
                df_group.index[-self.prediction_length :],
                quantiles=quantiles,
            )

        return df_predictions_dict, df_item_metrics_dict, df_agg_metrics

    def predict(
        self,
        group_df_dict: Dict[Tuple[Any, ...], pd.DataFrame],
        quantiles: List[float] = [0.05, 0.5, 0.95],
    ) -> Dict[Tuple[Any, ...], pd.DataFrame]:
        """Make a prediction using the model.

        Args:
            group_df_dict: Dictionary containing the time series for each group.
            quantiles: Quantiles to predict.

        Raises:
            UntrainedModelException: If the model has not been trained/tuned before.

        Returns:
            Dictionary containing the predicted time series for each group.
        """
        if self.predictor is None:
            raise UntrainedModelException()

        if not self.has_dynamic_features:
            future_dates_dict = {
                group: pd.date_range(
                    df.index[-1], periods=self.prediction_length + 1, freq=self.freq
                )[1:]
                for group, df in group_df_dict.items()
            }
        else:
            future_dates_dict = {
                group: df.index[-self.prediction_length :]
                for group, df in group_df_dict.items()
            }

        df_predictions_dict = {}

        data = dataframe_to_list_dataset(
            group_df_dict,
            self.target_columns,
            self.freq_gluon,
            real_static_feature_dict=self.real_static_feature_dict,
            cat_static_feature_dict=self.cat_static_feature_dict,
            real_dynamic_feature_columns=self.real_dynamic_feature_columns,
            cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
            group_label_dict=self.group_label_dict,
            prediction_length=self.prediction_length,
            training=False,
        )

        forecast_list = self.predictor.predict(data)
        for group, forecast in zip(group_df_dict.keys(), forecast_list):
            df_predictions_dict[group] = forecast_to_dataframe(
                forecast,
                self.target_columns,
                future_dates_dict[group],
                quantiles=quantiles,
            )

        return df_predictions_dict
