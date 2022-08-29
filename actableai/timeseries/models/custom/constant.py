import numpy as np

from gluonts.dataset import DataEntry
from gluonts.dataset.util import forecast_start
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import RepresentablePredictor, FallbackPredictor


class ConstantValuePredictor(RepresentablePredictor, FallbackPredictor):
    """
    A Predictor that always produces the same value as forecast. Inspired from the
    ConstantValuePredictor form GluonTS, just adding multivariate from it.
    """

    def __init__(
        self,
        prediction_length: int,
        value: float = 0.0,
        num_samples: int = 1,
        target_dim: int = 1,
    ):
        """ConstantValuePredictor Constructor.

        Args:
            value: The value to use as forecast.
            prediction_length: Prediction horizon.
            num_samples: Number of samples.
            target_dim: Target dimension of the forecast.
        """
        super().__init__(prediction_length=prediction_length)
        self.value = value
        self.num_samples = num_samples
        self.target_dim = target_dim

    def predict_item(self, item: DataEntry) -> SampleForecast:
        """Predict one item.

        Args:
            item: Item to make the prediction for.

        Returns:
            Sample forecast.
        """
        if self.target_dim == 1:
            samples_shape = self.num_samples, self.prediction_length
        else:
            samples_shape = self.num_samples, self.prediction_length, self.target_dim

        samples = np.full(samples_shape, self.value)
        return SampleForecast(
            samples=samples,
            start_date=forecast_start(item),
            item_id=item.get("item_id"),
        )
