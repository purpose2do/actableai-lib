from functools import partial

import numpy as np
import pandas as pd

from typing import Union, Optional, List, Callable

from gluonts.model import Forecast


class AAITimeSeriesForecast(Forecast):
    """Custom wrapper around GluonTS Forecast."""

    def __init__(
        self,
        forecast: Forecast,
        transformation_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """AAITimeSeriesForecast Constructor.

        Args:
            forecast: Underlying GluonTS forecast.
            transformation_func: Function that will be applied before returning a
                forecast. The function takes a numpy array and returns the transformed
                numpy array.
        """

        self.forecast = forecast

        self.start_date = self.forecast.start_date
        self.item_id = (
            self.forecast.item_id if hasattr(self.forecast, "item_id") else None
        )
        self.info = self.forecast.info if hasattr(self.forecast, "info") else None
        self.prediction_length = self.forecast.prediction_length
        self._index = self.forecast._index

        self.transformation_func = transformation_func
        self._transformation_func = None

        if self.transformation_func is None:
            self._transformation_func = lambda array: array
        else:
            self._transformation_func = partial(
                self.transformation_func, start_date=self.start_date
            )

    @property
    def mean(self) -> np.ndarray:
        """Mean of the forecast.

        Returns:
            Forecast's mean.
        """
        return self._transformation_func(self.forecast.mean)

    def quantile(self, q: Union[float, str]) -> np.ndarray:
        """Computes a quantile from the predicted distribution.

        Args:
            q: Quantile to compute.

        Returns:
            Value of the quantile across the prediction range.
        """

        result = self.forecast.quantile(q)
        result = self._transformation_func(result)

        return result

    def to_dataframe(
        self,
        target_columns: List[str],
        date_list: List[pd.datetime],
        quantiles: List[float] = [0.05, 0.5, 0.95],
    ) -> pd.DataFrame:
        """Convert GluonTS forecast to pandas DataFrame.

        Args:
            target_columns: List of columns to forecast.
            date_list: List of datetime forecasted.
            quantiles: List of quantiles to forecast.

        Returns:
            Forecasted values as pandas DataFrame.
        """
        prediction_length = self.prediction_length

        quantiles_values_dict = {
            quantile: self.quantile(quantile).astype(float) for quantile in quantiles
        }

        if len(target_columns) <= 1:
            for quantile in quantiles_values_dict.keys():
                quantiles_values_dict[quantile] = quantiles_values_dict[
                    quantile
                ].reshape(prediction_length, 1)

        return pd.concat(
            [
                pd.DataFrame(
                    {
                        "target": [target_column] * prediction_length,
                        "date": date_list,
                        **{
                            str(quantile): quantiles_values[:, index]
                            for quantile, quantiles_values in quantiles_values_dict.items()
                        },
                    }
                )
                for index, target_column in enumerate(target_columns)
            ],
            ignore_index=True,
        )

    def dim(self) -> int:
        """Returns the dimensionality of the forecast object.

        Returns:
            Forecast's dimension.
        """
        return self.forecast.dim()

    def copy_dim(self, dim: int) -> "AAITimeSeriesForecast":
        """Returns a new Forecast object with only the selected sub-dimension.

        Args:
            dim: The returned forecast object will only represent this dimension.

        Returns:
            Selected sub-forecast.
        """
        return AAITimeSeriesForecast(
            forecast=self.forecast.copy_dim(dim),
            transformation_func=self.transformation_func,
        )

    def copy_aggregate(self, agg_fun: Callable) -> "AAITimeSeriesForecast":
        """Returns a new Forecast object with a time series aggregated over the
            dimension axis.

        Args:
            agg_fun: Aggregation function that defines the aggregation operation
                (typically mean or sum).

        Returns:
            Selected sub-forecast.
        """
        return AAITimeSeriesForecast(
            forecast=self.forecast.copy_aggregate(agg_fun),
            transformation_func=self.transformation_func,
        )
