from typing import Tuple, Iterable, Optional, Union

import pandas as pd
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.predictor import Predictor

from actableai.timeseries.dataset import AAITimeSeriesDataset
from actableai.timeseries.forecast import AAITimeSeriesForecast
from actableai.timeseries.transform.base import Transformation
from actableai.timeseries.transform.identity import Identity


class AAITimeSeriesPredictor:
    """Custom Wrapper around GluonTS Predictor."""

    def __init__(
        self,
        predictor: Predictor,
        transformation: Optional[Transformation] = None,
    ):
        """AAITimeSeriesPredictor Constructor.

        Args:
            predictor: Underlying GluonTS predictor.
            transformation: Transformation to apply to the predictor.
        """
        self.predictor = predictor
        self.transformation = transformation

        if self.transformation is None:
            self.transformation = Identity()

    def make_evaluation_predictions(
        self, dataset: AAITimeSeriesDataset, num_samples: int
    ) -> Tuple[
        Iterable[AAITimeSeriesForecast], Iterable[Union[pd.DataFrame, pd.Series]]
    ]:
        """Wrapper around the GluonTS `make_evaluation_predictions` function.

        Args:
            dataset: Data used for evaluation.
            num_samples: Number of samples to draw on the model when evaluating.

        Returns:
            - Iterator containing the evaluation forecasts.
            - Iterator containing the original sampled time series.
        """
        dataset.training = True

        self.transformation.setup(dataset)
        dataset = self.transformation.transform(dataset)

        forecast_it, ts_it = make_evaluation_predictions(
            dataset, self.predictor, num_samples
        )

        forecast_it = self.transformation.revert_forecasts(forecast_it)
        ts_it = self.transformation.revert_time_series(ts_it)

        forecast_list = []
        for forecast in forecast_it:
            if isinstance(forecast, AAITimeSeriesForecast):
                forecast_list.append(forecast)
            else:
                forecast_list.append(AAITimeSeriesForecast(forecast=forecast))

        return forecast_list, ts_it

    def predict(
        self, dataset: AAITimeSeriesDataset, **kwargs
    ) -> Iterable[AAITimeSeriesForecast]:
        """Run prediction.

        Args:
            dataset: Data used for prediction.

        Returns:
            Predictions results
        """
        dataset.training = False

        self.transformation.setup(dataset)
        dataset = self.transformation.transform(dataset)

        forecast_it = self.predictor.predict(dataset, **kwargs)
        forecast_it = self.transformation.revert_forecasts(forecast_it)

        for forecast in forecast_it:
            if isinstance(forecast, AAITimeSeriesForecast):
                yield forecast
            else:
                yield AAITimeSeriesForecast(forecast=forecast)
