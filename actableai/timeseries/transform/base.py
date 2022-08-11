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
    """
    FIXME
    Base class for all Transformations.
    A Transformation processes works on a stream (iterator) of dictionaries.
    """

    def __init__(self):
        """
        TODO write documentation
        """
        self.dataset = None

    def setup(self, dataset: AAITimeSeriesDataset):
        """
        TODO write documentation
        """
        self.dataset = dataset

    @property
    def group_list(self):
        """
        TODO write documentation
        """
        if self.dataset is None:
            raise ValueError("transformation is not set up")
        return self.dataset.group_list

    @abc.abstractmethod
    def transform(self, data_it: Iterable[DataEntry]) -> Iterable[DataEntry]:
        """
        TODO write documentation
        """
        raise NotImplementedError()

    def revert_forecasts(self, forecast_it: Iterable[Forecast]) -> Iterable[Forecast]:
        """
        TODO write documentation
        """
        return forecast_it

    def revert_time_series(
        self, data_it: Iterable[Union[pd.DataFrame, pd.Series]]
    ) -> Iterable[Union[pd.DataFrame, pd.Series]]:
        """
        TODO write documentation
        """
        return data_it

    def chain(self, other: "Transformation") -> "Chain":
        """
        TODO write documentation
        """
        return Chain([self, other])

    def __add__(self, other: "Transformation") -> "Chain":
        """
        TODO write documentation
        """
        return self.chain(other)


class Chain(Transformation):
    """
    FIXME
    Chain multiple transformations together.
    """

    def __init__(self, transformations: List[Transformation]) -> None:
        """
        TODO write documentation
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
        super().setup(dataset)

        for transformation in self.transformations:
            transformation.setup(dataset)

    def transform(self, data_it: Iterable[DataEntry]) -> Iterable[DataEntry]:
        """
        TODO write documentation
        """
        tmp_data = data_it
        for transformation in self.transformations:
            tmp_data = transformation.transform(tmp_data)

        return tmp_data

    def revert_forecasts(self, forecast_it: Iterable[Forecast]) -> Iterable[Forecast]:
        """
        TODO write documentation
        """
        tmp_forecast = forecast_it
        for transformation in reversed(self.transformations):
            tmp_forecast = transformation.revert_forecasts(tmp_forecast)

        return tmp_forecast

    def revert_time_series(
        self, data_it: Iterable[Union[pd.DataFrame, pd.Series]]
    ) -> Iterable[Union[pd.DataFrame, pd.Series]]:
        """
        TODO write documentation
        """
        tmp_data = data_it
        for transformation in reversed(self.transformations):
            tmp_data = transformation.revert_time_series(tmp_data)

        return tmp_data


class MapTransformation(Transformation):
    """
    FIXME
    Base class for Transformations that returns exactly one result per input in
    the stream.
    """

    def transform(self, data_it: Iterable[DataEntry]) -> Iterable[DataEntry]:
        """
        TODO write documentation
        """
        return [
            self.map_transform(data, group)
            for data, group in zip(data_it, self.group_list)
        ]

    @abc.abstractmethod
    def map_transform(self, data: DataEntry, group: Tuple[Any, ...]) -> DataEntry:
        """
        TODO write documentation
        """
        raise NotImplementedError()

    def revert_forecasts(self, forecast_it: Iterable[Forecast]) -> Iterable[Forecast]:
        """
        TODO write documentation
        """
        for forecast, group in zip(forecast_it, self.group_list):
            yield self.map_revert_forecast(forecast, group)

    def map_revert_forecast(
        self, forecast: Forecast, group: Tuple[Any, ...]
    ) -> Forecast:
        """
        TODO write documentation
        """
        return forecast

    def revert_time_series(
        self, data_it: Iterable[Union[pd.DataFrame, pd.Series]]
    ) -> Iterable[Union[pd.DataFrame, pd.Series]]:
        """
        TODO write documentation
        """
        return [
            self.map_revert_time_series(data, group)
            for data, group in zip(data_it, self.group_list)
        ]

    def map_revert_time_series(
        self, data: Union[pd.DataFrame, pd.Series], group: Tuple[Any, ...]
    ):
        """
        TODO write documentation
        """
        return data


class ArrayTransformation(MapTransformation):
    """
    TODO write documentation
    """

    def map_transform(self, data: DataEntry, group: Tuple[Any, ...]) -> DataEntry:
        """
        TODO write documentation
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
        """
        TODO write documentation
        """
        raise NotImplementedError()

    def map_revert_forecast(
        self, forecast: Forecast, group: Tuple[Any, ...]
    ) -> Forecast:
        """
        TODO write documentation
        """
        return AAITimeSeriesForecast(
            forecast=forecast,
            transformation_func=partial(self.revert_array, group=group),
        )

    def map_revert_time_series(
        self, data: Union[pd.DataFrame, pd.Series], group: Tuple[Any, ...]
    ):
        """
        TODO write documentation
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
        """
        TODO write documentation
        """
        return array
