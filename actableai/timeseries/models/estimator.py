from typing import Optional

from gluonts.model.estimator import Estimator

from actableai.timeseries.dataset import AAITimeSeriesDataset
from actableai.timeseries.models.predictor import AAITimeSeriesPredictor
from actableai.timeseries.transform.base import Transformation
from actableai.timeseries.transform.identity import Identity


class AAITimeSeriesEstimator:
    """Custom Wrapper around GluonTS Estimator."""

    def __init__(
        self,
        estimator: Estimator,
        transformation: Optional[Transformation] = None,
    ):
        """AAITimeSeriesEstimator Constructor.

        Args:
            estimator: Underlying GluonTS estimator.
            transformation: Transformation to apply to the estimator.
        """
        self.estimator = estimator
        self.transformation = transformation

        if self.transformation is None:
            self.transformation = Identity()

    def train(
        self,
        dataset: AAITimeSeriesDataset,
    ) -> AAITimeSeriesPredictor:
        """Train estimator.

        Args:
            dataset: Training data.

        Returns:
            GluonTS trained predictor.
        """
        dataset.training = True

        self.transformation.setup(dataset)
        dataset = self.transformation.transform(dataset)

        predictor = self.estimator.train(dataset)

        return AAITimeSeriesPredictor(predictor, transformation=self.transformation)
