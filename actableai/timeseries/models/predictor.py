import pandas as pd
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.forecast import Forecast
from gluonts.model.predictor import Predictor
from typing import Iterator, Tuple

from actableai.timeseries.dataset import AAITimeSeriesDataset


class AAITimeSeriesPredictor:
    """Custom Wrapper around GluonTS Predictor."""

    def __init__(
        self,
        predictor: Predictor,
        keep_feat_static_real: bool = True,
        keep_feat_static_cat: bool = True,
        keep_feat_dynamic_real: bool = True,
        keep_feat_dynamic_cat: bool = True,
    ):
        """AAITimeSeriesPredictor Constructor.

        Args:
            predictor: Underlying GluonTS predictor.
            keep_feat_static_real: If False the real static features will be filtered
                out.
            keep_feat_static_cat: If False the categorical static features will be
                filtered out.
            keep_feat_dynamic_real: If False the real dynamic features will be filtered
                out.
            keep_feat_dynamic_cat: If False the categorical dynamic features will be
                filtered out.
        """
        self.predictor = predictor

        self.keep_feat_static_real = keep_feat_static_real
        self.keep_feat_static_cat = keep_feat_static_cat
        self.keep_feat_dynamic_real = keep_feat_dynamic_real
        self.keep_feat_dynamic_cat = keep_feat_dynamic_cat

    def make_evaluation_predictions(
        self, dataset: AAITimeSeriesDataset, num_samples: int
    ) -> Tuple[Iterator[Forecast], Iterator[pd.Series]]:
        """Wrapper around the GluonTS `make_evaluation_predictions` function.

        Args:
            dataset: Data used for evaluation.
            num_samples: Number of samples to draw on the model when evaluating.

        Returns:
            - Iterator containing the evaluation forecasts.
            - Iterator containing the original sampled time series.
        """
        dataset = dataset.clean_features(
            self.keep_feat_static_real,
            self.keep_feat_static_cat,
            self.keep_feat_dynamic_real,
            self.keep_feat_dynamic_cat,
        )
        dataset.training = True

        return make_evaluation_predictions(dataset, self.predictor, num_samples)

    def predict(self, dataset: AAITimeSeriesDataset, **kwargs) -> Iterator[Forecast]:
        """Run prediction.

        Args:
            dataset: Data used for prediction.

        Returns:
            Predictions results
        """
        dataset = dataset.clean_features(
            self.keep_feat_static_real,
            self.keep_feat_static_cat,
            self.keep_feat_dynamic_real,
            self.keep_feat_dynamic_cat,
        )
        dataset.training = False

        return self.predictor.predict(dataset, **kwargs)
