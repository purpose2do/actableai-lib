import abc
from copy import deepcopy
from functools import partial
from typing import List, Iterable, Union, Tuple, Any

import numpy as np
import pandas as pd

from gluonts.dataset import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.model import Forecast

from actableai.timeseries.dataset import AAITimeSeriesDataset
from actableai.timeseries.forecast import AAITimeSeriesForecast


class Transformation(metaclass=abc.ABCMeta):
    """Base class for all Transformations."""

    def __init__(self):
        """Transformation constructor."""
        self.dataset = None

    def setup(self, dataset: AAITimeSeriesDataset):
        """Set up the transformation with a dataset.

        Args:
            dataset: Dataset to set up the transformation with.
        """
        self.dataset = dataset

    @property
    def group_list(self) -> List[Tuple[Any, ...]]:
        """Returns the list of group associated with the transformation.

        Returns:
            The list of group.
        """
        if self.dataset is None:
            raise ValueError("transformation is not set up")
        return self.dataset.group_list

    @abc.abstractmethod
    def transform(self, data_it: Iterable[DataEntry]) -> Iterable[DataEntry]:
        """Transform data entries.

        Args:
            data_it: Iterable object of data entries.

        Returns:
            Iterable object of transformed data entries.
        """
        raise NotImplementedError()

    def revert_forecasts(self, forecast_it: Iterable[Forecast]) -> Iterable[Forecast]:
        """Revert a transformation on forecasts.

        Args:
            forecast_it: Iterable object of forecasts.

        Returns:
            Iterable object of transformed forecasts.
        """
        return forecast_it

    def revert_time_series(
        self, data_it: Iterable[Union[pd.DataFrame, pd.Series]]
    ) -> Iterable[Union[pd.DataFrame, pd.Series]]:
        """Revert a transformation on time series.

        Args:
            data_it: Iterable object of time series.

        Returns:
            Iterable object of transformed time series.
        """
        return data_it

    def chain(self, other: "Transformation") -> "Chain":
        """Chain transformation with the current transformation.

        Args:
            other: Other transformation to apply.

        Returns:
            The chain of transformation containing the current one and the other.
        """
        return Chain([self, other])

    def __add__(self, other: "Transformation") -> "Chain":
        """Chain transformation with the current transformation.

        Args:
            other: Other transformation to apply.

        Returns:
            The chain of transformation containing the current one and the other.
        """
        return self.chain(other)


class Chain(Transformation):
    """Chain multiple transformations together."""

    def __init__(self, transformations: List[Transformation]):
        """Chain transformation constructor.

        Args:
            transformations: List of transformation to chain.
        """
        super().__init__()

        self.transformations: List[Transformation] = []

        for transformation in transformations:
            # flatten chains
            if isinstance(transformation, Chain):
                self.transformations.extend(transformation.transformations)
            else:
                self.transformations.append(transformation)

    def setup(self, dataset: AAITimeSeriesDataset):
        """Set up the transformation with a dataset.

        Args:
            dataset: Dataset to set up the transformation with.
        """
        super().setup(dataset)

        for transformation in self.transformations:
            transformation.setup(dataset)

    def transform(self, data_it: Iterable[DataEntry]) -> Iterable[DataEntry]:
        """Transform data entries.

        Args:
            data_it: Iterable object of data entries.

        Returns:
            Iterable object of transformed data entries.
        """
        tmp_data = data_it
        for transformation in self.transformations:
            tmp_data = transformation.transform(tmp_data)

        return tmp_data

    def revert_forecasts(self, forecast_it: Iterable[Forecast]) -> Iterable[Forecast]:
        """Revert a transformation on forecasts.

        Args:
            forecast_it: Iterable object of forecasts.

        Returns:
            Iterable object of transformed forecasts.
        """
        tmp_forecast = forecast_it
        for transformation in reversed(self.transformations):
            tmp_forecast = transformation.revert_forecasts(tmp_forecast)

        return tmp_forecast

    def revert_time_series(
        self, data_it: Iterable[Union[pd.DataFrame, pd.Series]]
    ) -> Iterable[Union[pd.DataFrame, pd.Series]]:
        """Revert a transformation on time series.

        Args:
            data_it: Iterable object of time series.

        Returns:
            Iterable object of transformed time series.
        """
        tmp_data = data_it
        for transformation in reversed(self.transformations):
            tmp_data = transformation.revert_time_series(tmp_data)

        return tmp_data


class MapTransformation(Transformation):
    """
    Base class for Transformations that returns exactly one result per input in the
        stream.
    """

    def transform(self, data_it: Iterable[DataEntry]) -> Iterable[DataEntry]:
        """Transform data entries.

        Args:
            data_it: Iterable object of data entries.

        Returns:
            Iterable object of transformed data entries.
        """
        return [
            self.map_transform(data, group)
            for data, group in zip(data_it, self.group_list)
        ]

    @abc.abstractmethod
    def map_transform(self, data: DataEntry, group: Tuple[Any, ...]) -> DataEntry:
        """Transform a data entry.

        Args:
            data: Data entry to revert.
            group: Data entry's group.

        Returns:
            The transformed data entry.
        """
        raise NotImplementedError()

    def revert_forecasts(self, forecast_it: Iterable[Forecast]) -> Iterable[Forecast]:
        """Revert a transformation on forecasts.

        Args:
            forecast_it: Iterable object of forecasts.

        Returns:
            Iterable object of transformed forecasts.
        """
        for forecast, group in zip(forecast_it, self.group_list):
            yield self.map_revert_forecast(forecast, group)

    def map_revert_forecast(
        self, forecast: Forecast, group: Tuple[Any, ...]
    ) -> Forecast:
        """Revert a transformation on a forecast.

        Args:
            forecast: Forecast to revert.
            group: Forecast's group.

        Returns:
            The transformed forecast.
        """
        return forecast

    def revert_time_series(
        self, data_it: Iterable[Union[pd.DataFrame, pd.Series]]
    ) -> Iterable[Union[pd.DataFrame, pd.Series]]:
        """Revert a transformation on time series.

        Args:
            data_it: Iterable object of time series.

        Returns:
            Iterable object of transformed time series.
        """
        return [
            self.map_revert_time_series(data, group)
            for data, group in zip(data_it, self.group_list)
        ]

    def map_revert_time_series(
        self, data: Union[pd.DataFrame, pd.Series], group: Tuple[Any, ...]
    ) -> Union[pd.DataFrame, pd.Series]:
        """Revert a transformation on a time series.

        Args:
            data: Time series to revert.
            group: Time series group.

        Returns:
            The transformed time series.
        """
        return data


class ArrayTransformation(MapTransformation):
    """Base class for Transformations than are applied on numpy array."""

    def map_transform(self, data: DataEntry, group: Tuple[Any, ...]) -> DataEntry:
        """Transform a data entry.

        Args:
            data: Data entry to revert.
            group: Data entry's group.

        Returns:
            The transformed data entry.
        """
        data = deepcopy(data)
        data[FieldName.TARGET] = self.transform_array(
            array=data[FieldName.TARGET],
            start_date=data[FieldName.START],
            group=group,
        )
        return data

    @abc.abstractmethod
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
        raise NotImplementedError()

    def map_revert_forecast(
        self, forecast: Forecast, group: Tuple[Any, ...]
    ) -> Forecast:
        """Revert a transformation on a forecast.

        Args:
            forecast: Forecast to revert.
            group: Forecast's group.

        Returns:
            The transformed forecast.
        """
        return AAITimeSeriesForecast(
            forecast=forecast,
            transformation_func=partial(self.revert_array, group=group),
        )

    def map_revert_time_series(
        self, data: Union[pd.DataFrame, pd.Series], group: Tuple[Any, ...]
    ) -> Union[pd.DataFrame, pd.Series]:
        """Revert a transformation on a time series.

        Args:
            data: Time series to revert.
            group: Time series group.

        Returns:
            The transformed time series.
        """
        data = deepcopy(data)
        if isinstance(data, pd.DataFrame) and len(data.columns) == 1:
            data = data[data.columns[0]]

        reverted_data = self.revert_array(
            array=data.to_numpy(), start_date=data.index[0], group=group
        )

        if isinstance(data, pd.Series):
            return pd.Series(data=reverted_data, index=data.index, name=data.name)

        return pd.DataFrame(data=reverted_data, index=data.index, columns=data.columns)

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
        return array
