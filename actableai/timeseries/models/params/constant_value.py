from typing import Tuple, Union, Dict, Any

from actableai.timeseries.models.custom.constant import ConstantValuePredictor
from actableai.timeseries.models.params.base import BaseParams
from actableai.timeseries.models.predictor import AAITimeSeriesPredictor


class ConstantValueParams(BaseParams):
    """Parameters class for the Constant Value Model."""

    def __init__(
        self,
        value: Union[Tuple[int, int], int] = (0, 100),
        target_dim: int = 1,
    ):
        """ConstantValueParams Constructor.

        Args:
            value: Value to return, if tuple it represents minimum and maximum
                (excluded) value.
            target_dim: Dimension of the target.
        """

        super().__init__(
            model_name="ConstantValue",
            is_multivariate_model=target_dim > 1,
            has_estimator=False,
            handle_feat_static_real=False,
            handle_feat_static_cat=False,
            handle_feat_dynamic_real=False,
            handle_feat_dynamic_cat=False,
        )

        self.value = value
        self.target_dim = target_dim

    def tune_config(self) -> Dict[str, Any]:
        """Select parameters in the pre-defined hyperparameter space.

        Returns:
            Selected parameters.
        """
        return {"value": self._uniform("value", self.value)}

    def build_predictor(
        self, *, prediction_length: int, params: Dict[str, Any], **kwargs
    ) -> AAITimeSeriesPredictor:
        """Build a predictor from the underlying model using selected parameters.

        Args:
            prediction_length: Length of the prediction that will be forecasted.
            params: Selected parameters from the hyperparameter space.
            kwargs: Ignored arguments.

        Returns:
            Built predictor.
        """

        return self._create_predictor(
            ConstantValuePredictor(
                value=params.get("value", self.value),
                prediction_length=prediction_length,
                target_dim=self.target_dim,
            )
        )
