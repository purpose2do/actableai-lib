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
from typing import Dict, List, Optional, Tuple, Any

from actableai.exceptions.timeseries import UntrainedModelException
from actableai.timeseries.dataset import AAITimeSeriesDataset
from actableai.timeseries.models.base import AAITimeSeriesBaseModel
from actableai.timeseries.models.params.base import BaseParams
from actableai.timeseries.models.predictor import AAITimeSeriesPredictor
from actableai.timeseries.models.evaluator import AAITimeSeriesEvaluator


class AAITimeSeriesSingleModel(AAITimeSeriesBaseModel):
    """Simple Time Series Model,"""

    def __init__(self, prediction_length: int):
        """AAITimeSeriesBaseModel Constructor.

        Args:
            prediction_length: Length of the prediction to forecast.
        """
        super().__init__(prediction_length)

        self.predictor = None
        self.best_params = None

        self.model_params_dict = None
        self.distr_output = None

    @staticmethod
    def _create_predictor(
        model_params_dict: Dict[str, BaseParams],
        params: Dict[str, Any],
        dataset: AAITimeSeriesDataset,
        distr_output: DistributionOutput,
        prediction_length: int,
        mx_ctx: mx.Context,
    ) -> AAITimeSeriesPredictor:
        """Create and train a predictor.

        Args:
            model_params_dict: Dictionary containing the different model params for
                each model.
            params: Hyperparameter choose by the tuning.
            dataset: Data to use for training.
            distr_output: Distribution output to use.
            prediction_length: Length of the prediction that will be forecasted.
            mx_ctx: mxnet context.

        Returns:
            Trained predictor.
        """
        model_params_class = model_params_dict[params["model"]["model_name"]]

        if model_params_class.has_estimator:
            estimator = model_params_class.build_estimator(
                ctx=mx_ctx,
                freq=dataset.gluonts_freq,
                prediction_length=prediction_length,
                target_dim=len(dataset.target_columns),
                distr_output=distr_output,
                params=params["model"],
            )

            predictor = estimator.train(dataset=dataset)
        else:
            predictor = model_params_class.build_predictor(
                freq=dataset.gluonts_freq,
                prediction_length=prediction_length,
                params=params["model"],
            )

        return predictor

    @classmethod
    def _trainable(
        cls,
        params: Dict[str, Any],
        *,
        model_params_dict: Dict[str, BaseParams],
        train_dataset_partial: AAITimeSeriesDataset,
        tune_dataset: AAITimeSeriesDataset,
        loss: str,
        distr_output: DistributionOutput,
        prediction_length: int,
        use_ray: bool,
        mx_ctx: mx.Context,
    ) -> Optional[pd.DataFrame]:
        """Create, train, and evaluate a model with specific hyperparameter.

        Args:
            params: Hyperparameter choose by the tuning.
            model_params_dict: Dictionary containing the different model params for
                each model.
            train_dataset_partial: Data to use for training.
            tune_dataset: Data to use for tuning.
            loss: Loss to return.
            distr_output: Distribution output to use.
            prediction_length: Length of the prediction that will be forecasted.
            use_ray: Whether ray is used for tuning or not.
            mx_ctx: mxnet context.

        Returns:
            If `use_ray` is False return the loss. Else will report the loss to ray
            tune.
        """
        predictor = cls._create_predictor(
            model_params_dict,
            params,
            train_dataset_partial,
            distr_output,
            prediction_length,
            mx_ctx,
        )

        forecast_it, ts_it = predictor.make_evaluation_predictions(
            tune_dataset, num_samples=100
        )

        evaluator = AAITimeSeriesEvaluator(
            target_columns=train_dataset_partial.target_columns,
            quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
            num_workers=None,
        )
        _, df_agg_metrics = evaluator(ts_it, forecast_it, num_series=len(tune_dataset))

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
        train_dataset_partial: AAITimeSeriesDataset,
        tune_dataset: AAITimeSeriesDataset,
        loss: str,
        distr_output: DistributionOutput,
        prediction_length: int,
        use_ray: bool,
        mx_ctx: mx.Context,
    ) -> Dict[str, Any]:
        """Create, train, and evaluate a model with specific hyperparameter. Used by
            hyperopt.

        Args:
            params: Hyperparameter choose by the tuning.
            model_params_dict: Dictionary containing the different model params for
                each model.
            train_dataset_partial: Data to use for training.
            tune_dataset: Data to use for tuning.
            loss: Loss to return.
            distr_output: Distribution output to use.
            use_ray: Whether ray is used for tuning or not.
            mx_ctx: mxnet context.

        Returns:
            Dictionary containing the loss and the status.
        """
        df_agg_metrics = cls._trainable(
            params,
            model_params_dict=model_params_dict,
            train_dataset_partial=train_dataset_partial,
            tune_dataset=tune_dataset,
            loss=loss,
            distr_output=distr_output,
            prediction_length=prediction_length,
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
        dataset: AAITimeSeriesDataset,
        tune_samples: int,
        sampling_method: str = "random",
    ) -> Tuple[AAITimeSeriesDataset, AAITimeSeriesDataset, AAITimeSeriesDataset]:
        """Generate and split train and validation data for tuning.

        Args:
            dataset: Dataset containing the time series.
            tune_samples: Number of dataset samples to use when tuning.
            sampling_method: Method used when extracting the samples for the tuning
                ["random", "last"].

        Returns:
            - Training ListDataset.
            - Training ListDataset (partial without tuning).
            - Tuning ListDataset.
        """
        train_dataset = dataset

        train_dataset_partial = dataset.slice_data(
            slice_df=slice(-self.prediction_length - tune_samples)
        )

        tune_dataset_list = []
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

            tune_dataset_list += list(
                dataset.slice_data(slice_df=slice_function).dataframes.values()
            )

        tune_dataset = AAITimeSeriesDataset(
            dataframes=tune_dataset_list,
            target_columns=dataset.target_columns,
            freq=dataset.freq,
            prediction_length=dataset.prediction_length,
            feat_dynamic_real=dataset.feat_dynamic_real,
            feat_dynamic_cat=dataset.feat_dynamic_cat,
            feat_static_real=dataset.feat_static_real,
            feat_static_cat=dataset.feat_static_cat,
        )

        return train_dataset, train_dataset_partial, tune_dataset

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
        self.model_params_dict = {
            model_param_class.model_name: model_param_class
            for model_param_class in model_params
            if model_param_class is not None
        }

        # Split data
        (
            train_dataset,
            train_dataset_partial,
            tune_dataset,
        ) = self._generate_train_valid_data(
            dataset,
            tune_samples,
            sampling_method,
        )

        # Choose distribution output
        first_group_targets = train_dataset.dataframes[train_dataset.group_list[0]][
            train_dataset.target_columns
        ]
        if (first_group_targets >= 0).all(
            axis=None
        ) and first_group_targets in visions.Integer:
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
                train_dataset_partial=train_dataset_partial,
                tune_dataset=tune_dataset,
                loss=loss,
                distr_output=self.distr_output,
                prediction_length=self.prediction_length,
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
                train_dataset_partial=train_dataset_partial,
                tune_dataset=tune_dataset,
                loss=loss,
                distr_output=self.distr_output,
                prediction_length=self.prediction_length,
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
            str(dataset.target_columns)
            if len(dataset.target_columns) > 1
            else dataset.target_columns[0]
        )

        if fit_full:
            final_train_dataset = train_dataset
        else:
            final_train_dataset = train_dataset_partial

        # Create final model using best parameters
        self.predictor = self._create_predictor(
            model_params_dict=self.model_params_dict,
            params=self.best_params,
            dataset=final_train_dataset,
            distr_output=self.distr_output,
            prediction_length=self.prediction_length,
            mx_ctx=mx_ctx,
        )

        return time.time() - start + trials_time_total, df_leaderboard

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
        if self.predictor is None:
            raise UntrainedModelException()

        self.predictor = self._create_predictor(
            model_params_dict=self.model_params_dict,
            params=self.best_params,
            dataset=dataset,
            distr_output=self.distr_output,
            prediction_length=self.prediction_length,
            mx_ctx=mx_ctx,
        )

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
        if self.predictor is None:
            raise UntrainedModelException()

        df_predictions_dict = {}

        forecast_it, ts_it = self.predictor.make_evaluation_predictions(
            dataset, num_samples
        )

        ts_list = list(ts_it)
        forecast_list = list(forecast_it)

        # Create custom metric function
        rmse = lambda target, forecast: np.sqrt(np.mean(np.square(target - forecast)))

        # Evaluate
        evaluator = AAITimeSeriesEvaluator(
            target_columns=dataset.target_columns,
            group_list=dataset.group_list,
            quantiles=quantiles,
            num_workers=num_workers,
            custom_eval_fn={"custom_RMSE": [rmse, "mean", "median"]},
        )
        df_item_metrics_dict, df_agg_metrics = evaluator(
            ts_list, forecast_list, num_series=len(dataset)
        )

        # Add predictions
        for (group, df_group), forecast in zip(
            dataset.dataframes.items(), forecast_list
        ):
            df_predictions_dict[group] = forecast.to_dataframe(
                dataset.target_columns,
                df_group.index[-self.prediction_length :],
                quantiles=quantiles,
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
        if self.predictor is None:
            raise UntrainedModelException()

        if not dataset.has_dynamic_features:
            future_dates_dict = {
                group: pd.date_range(
                    df.index[-1], periods=self.prediction_length + 1, freq=dataset.freq
                )[1:]
                for group, df in dataset.dataframes.items()
            }
        else:
            future_dates_dict = {
                group: df.index[-self.prediction_length :]
                for group, df in dataset.dataframes.items()
            }

        df_predictions_dict = {}

        forecast_list = self.predictor.predict(dataset)

        for group, forecast in zip(dataset.group_list, forecast_list):
            df_predictions_dict[group] = forecast.to_dataframe(
                dataset.target_columns,
                future_dates_dict[group],
                quantiles=quantiles,
            )

        return df_predictions_dict
