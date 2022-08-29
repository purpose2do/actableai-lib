import mxnet as mx
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional

from actableai.timeseries.dataset import AAITimeSeriesDataset
from actableai.timeseries.models.params.base import BaseParams


class AAITimeSeriesBaseModel(ABC):
    """Time Series Model interface."""

    def __init__(self, prediction_length: int):
        """AAITimeSeriesBaseModel Constructor.

        Args:
            prediction_length: Length of the prediction to forecast.
        """
        self.prediction_length = prediction_length

    @abstractmethod
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
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()
