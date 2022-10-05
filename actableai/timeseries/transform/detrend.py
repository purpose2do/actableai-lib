from typing import Tuple, Any, Iterable

import numpy as np
import pandas as pd
from gluonts.dataset import DataEntry
from gluonts.dataset.field_names import FieldName
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

from actableai.timeseries.transform.base import ArrayTransformation


class Detrend(ArrayTransformation):
    """Detrend transformation using model fitting."""

    def __init__(self):
        """Detrend constructor."""
        super().__init__()

        self._trend_models = None
        self._trend_start_date = None

    @staticmethod
    def _train_trend_model(data: DataEntry) -> MultiOutputRegressor:
        """Train model representing the trend.

        Args:
            FIXME
            dataset: Dataset to use for training.
            group: Group to use when selecting the time series from the dataset.

        Returns:
            The trained model.
        """
        df = pd.DataFrame(data[FieldName.TARGET].T)
        df = df.dropna()

        X = np.arange(df.shape[0]).reshape(-1, 1)
        y = df.to_numpy()

        model = MultiOutputRegressor(LinearRegression(), n_jobs=1)
        model.fit(X, y)

        return model

    def _predict_trend(
        self, group: Tuple[Any, ...], start_date: pd.Period, prediction_length: int
    ) -> np.ndarray:
        """Predict trend using trained model.

        Args:
            group: Group to predict the trend for.
            start_date: Starting date of the prediction.
            prediction_length: Number of periods to predict.

        Returns:
            The predicted trend.
        """
        periods = (
            start_date - pd.Period(self._trend_start_date[group], freq=start_date.freq)
        ).n

        X = (np.arange(prediction_length) + periods).reshape(-1, 1)
        return self._trend_models[group].predict(X)

    def _setup_data(self, data_it: Iterable[DataEntry]):
        """Set up the transformation with a dataset.

        Args:
            dataset: Dataset to set up the transformation with.
            FIXME
        """
        super()._setup_data(data_it)

        self._trend_models = {
            group: self._train_trend_model(data)
            for data, group in zip(data_it, self.group_list)
        }

        self._trend_start_date = {
            group: data[FieldName.START]
            for data, group in zip(data_it, self.group_list)
        }

    def transform_array(
        self, array: np.ndarray, start_date: pd.Period, group: Tuple[Any, ...]
    ) -> np.ndarray:
        """Transform an array.

        Args:
            array: Array to transform.
            start_date: Starting date of the array (in the time series context).
            group: Array's group.

        Returns:
            The transformed array.
        """
        trend = self._predict_trend(group, start_date, array.shape[-1])
        return array - trend.T

    def revert_array(
        self, array: np.ndarray, start_date: pd.Period, group: Tuple[Any, ...]
    ) -> np.ndarray:
        """Revert a transformation on an array.

        Args:
            array: Array to revert.
            start_date: Starting date of the array (in the time series context).
            group: Array's group.

        Returns:
            The transformed array.
        """
        trend = self._predict_trend(group, start_date, array.shape[-1])
        return array + trend.T
