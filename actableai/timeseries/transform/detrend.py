from copy import deepcopy
from typing import Tuple, Any

import numpy as np
import pandas as pd
from gluonts.dataset import DataEntry
from gluonts.dataset.field_names import FieldName
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

from actableai.timeseries.dataset import AAITimeSeriesDataset
from actableai.timeseries.transform.base import ArrayTransformation


class DetrendDiff(ArrayTransformation):
    """Detrend transformation using differentiation."""

    def __init__(self, n_differencing: int = 1):
        """DetrendDiff constructor.

        Args:
            n_differencing: Number of chained differentiation to run.
        """
        super().__init__()
        self.n_differencing = n_differencing

        self._intermediate_df_dict = None

    def setup(self, dataset: AAITimeSeriesDataset):
        """Set up the transformation with a dataset.

        Args:
            dataset: Dataset to set up the transformation with.
        """
        super().setup(dataset)

        self._intermediate_df_dict = {
            group: [pd.DataFrame() for _ in range(self.n_differencing)]
            for group in self.group_list
        }

    def map_transform(self, data: DataEntry, group: Tuple[Any, ...]) -> DataEntry:
        """Transform a data entry.

        Args:
            data: Data entry to revert.
            group: Data entry's group.

        Returns:
            The transformed data entry.
        """
        transformed_data = super().map_transform(data, group)

        fields = [FieldName.FEAT_DYNAMIC_REAL, FieldName.FEAT_DYNAMIC_CAT]

        for field in fields:
            if field in transformed_data:
                transformed_data[field] = transformed_data[field][
                    :, : -self.n_differencing
                ]
        transformed_data[FieldName.START] += self.n_differencing

        return transformed_data

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
        date_range = pd.period_range(
            start=start_date, freq=start_date.freq, periods=array.shape[-1]
        )

        univariate = len(array.shape) == 1

        if univariate:
            array = array.reshape(1, -1)

        # FIXME check that the array is big enough
        for diff_index in range(self.n_differencing):
            intermediate_df = pd.DataFrame(
                array.T.copy(),
                index=date_range[diff_index:],
            )

            self._intermediate_df_dict[group][diff_index] = pd.concat(
                [self._intermediate_df_dict[group][diff_index], intermediate_df], axis=0
            )

            for i in range(array.shape[1] - 1, 0, -1):
                array[:, i] = array[:, i] - array[:, i - 1]

            # Trim the array
            array = array[:, 1:]

        if univariate:
            array = array[0, :]

        return array

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
        prev_date = deepcopy(start_date)

        univariate = len(array.shape) == 1

        if univariate:
            array = array.reshape(1, -1)

        for diff_index in range(self.n_differencing - 1, -1, -1):
            prev_date -= 1

            # FIXME check if prev_date is valid index
            array = np.concatenate(
                (
                    self._intermediate_df_dict[group][diff_index]
                    .loc[prev_date]
                    .to_numpy()
                    .reshape(-1, 1),
                    array,
                ),
                axis=-1,
            )

            for i in range(1, array.shape[1]):
                array[:, i] = array[:, i] + array[:, i - 1]

            # Trim the array
            array = array[:, :-1]

        if univariate:
            array = array[0, :]

        return array


class Detrend(ArrayTransformation):
    """Detrend transformation using model fitting."""

    def __init__(self):
        """Detrend constructor."""
        super().__init__()

        self._trend_models = None
        self._trend_start_date = None

    @staticmethod
    def _train_trend_model(
        dataset: AAITimeSeriesDataset, group: Tuple[Any, ...]
    ) -> MultiOutputRegressor:
        """Train model representing the trend.

        Args:
            dataset: Dataset to use for training.
            group: Group to use when selecting the time series from the dataset.

        Returns:
            The trained model.
        """
        df = dataset.dataframes[group][dataset.target_columns]
        if dataset.has_dynamic_features:
            df = df.iloc[: -dataset.prediction_length]

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

    def setup(self, dataset: AAITimeSeriesDataset):
        """Set up the transformation with a dataset.

        Args:
            dataset: Dataset to set up the transformation with.
        """
        super().setup(dataset)

        self._trend_models = {
            group: self._train_trend_model(dataset, group)
            for group in dataset.group_list
        }

        self._trend_start_date = {
            group: dataset.dataframes[group].index[0] for group in dataset.group_list
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
        univariate = len(array.shape) == 1

        trend = self._predict_trend(group, start_date, array.shape[-1])
        if univariate:
            trend = trend[:, 0]
        else:
            trend = trend.T

        return array - trend

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
        univariate = len(array.shape) == 1

        trend = self._predict_trend(group, start_date, array.shape[-1])
        if univariate:
            trend = trend[:, 0]
        else:
            trend = trend.T

        return array + trend
