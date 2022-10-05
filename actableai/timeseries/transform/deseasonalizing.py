from typing import Tuple, Any, List, Iterable

import numpy as np
import pandas as pd
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName

from actableai.timeseries.transform.base import (
    ArrayTransformation,
    Chain,
)


class MultiDeseasonalizing(Chain):
    """
    TODO write documentation
    """

    def __init__(self):
        """
        TODO write documentation
        """
        super().__init__(transformations=[], is_flattenable=False)

    def _setup(
        self,
        data_it: Iterable[DataEntry],
        group_list: List[Tuple[Any, ...]],
        seasonal_periods: List[int],
    ):
        """
        TODO write documentation
        """
        self.transformations = []

        if seasonal_periods is not None and len(seasonal_periods) > 0:
            for seasonal_period in seasonal_periods:
                self.transformations.append(Deseasonalizing(seasonal_period))

        super()._setup(data_it, group_list, seasonal_periods)


class Deseasonalizing(ArrayTransformation):
    """
    TODO write documentation
    """

    def __init__(self, seasonal_period: int):
        """
        TODO write documentation
        """
        super().__init__()

        self.seasonal_period = seasonal_period

        self._seasonality = None
        self._seasonal_start_date = None

    def _train_seasonal_model(self, data: DataEntry):
        """
        TODO write documentation
        """

        df = pd.DataFrame(data[FieldName.TARGET].T)
        df = df.dropna()

        X = [i % self.seasonal_period for i in range(df.shape[0])]

        seasonality = []
        for col in df.columns:
            df_col_seasonality = pd.DataFrame()
            df_col_seasonality["index"] = X
            df_col_seasonality["val"] = df[col]

            df_col_seasonality = df_col_seasonality.groupby("index")["val"].mean()

            seasonality.append(df_col_seasonality.values)

        return seasonality

    def _predict_seasonality(
        self, group: Tuple[Any, ...], start_date: pd.Period, prediction_length: int
    ):
        """
        TODO write documentation
        """
        periods = (
            start_date
            - pd.Period(self._seasonal_start_date[group], freq=start_date.freq)
        ).n

        X = [(i + periods) % self.seasonal_period for i in range(prediction_length)]

        return np.array(
            [[seasonality[i] for i in X] for seasonality in self._seasonality[group]]
        )

    def _setup_data(self, data_it: Iterable[DataEntry]):
        """Set up the transformation with a dataset.

        Args:
            dataset: Dataset to set up the transformation with.
            FIXME
        """
        super()._setup_data(data_it)

        self._seasonality = {
            group: self._train_seasonal_model(data)
            for data, group in zip(data_it, self.group_list)
        }

        self._seasonal_start_date = {
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
        seasonality = self._predict_seasonality(group, start_date, array.shape[-1])
        return array - seasonality

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
        seasonality = self._predict_seasonality(group, start_date, array.shape[-1])
        return array + seasonality
