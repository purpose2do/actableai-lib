from functools import lru_cache
from typing import Dict, Any

from gluonts.model.rotbaum import TreeEstimator

from actableai.parameters.numeric import FloatRangeSpace
from actableai.parameters.options import OptionsSpace
from actableai.parameters.parameters import Parameters
from actableai.timeseries.models.estimator import AAITimeSeriesEstimator
from actableai.timeseries.models.params.base import BaseParams, Model
from actableai.timeseries.transform.deseasonalizing import MultiDeseasonalizing
from actableai.timeseries.transform.detrend import Detrend
from actableai.timeseries.transform.power_transformation import PowerTransformation


class TreePredictorParams(BaseParams):
    """Parameters class for Tree Predictor Model."""

    @staticmethod
    @lru_cache(maxsize=None)
    def get_hyperparameters() -> Parameters:
        """Returns the hyperparameters space of the model.

        Returns:
            The hyperparameters space.
        """

        parameters = [
            OptionsSpace[str](
                name="method",
                display_name="Method",
                # TODO add description
                description="description_method_todo",
                default=["QRX", "QuantileRegression"],
                options={
                    "QRX": {"display_name": "QRX", "value": "QRX"},
                    "QuantileRegression": {
                        "display_name": "Quantile Regression",
                        "value": "QuantileRegression",
                    },
                    "QRF": {"display_name": "QRF", "value": "QRF"},
                },
                # TODO check available options
            ),
            FloatRangeSpace(
                name="context_length_ratio",
                display_name="Context Length Ratio",
                description="Number of steps to unroll the RNN for before computing predictions. The Context Length is computed by multiplying this ratio with the Prediction Length.",
                default=(1, 2),
                min=1,
                # TODO check constraints
            ),
        ]

        return Parameters(
            name=Model.tree_predictor,
            display_name="Tree Predictor",
            parameters=parameters,
        )

    def __init__(
        self,
        hyperparameters: Dict = None,
        process_hyperparameters: bool = True,
    ):
        """TreePredictorParams Constructor.

        Args:
            hyperparameters: Dictionary representing the hyperparameters space.
            process_hyperparameters: If True the hyperparameters will be validated and
                processed (deactivate if they have already been validated).
        """
        super().__init__(
            model_name="TreePredictor",
            is_multivariate_model=False,
            has_estimator=True,
            handle_feat_static_real=True,
            handle_feat_static_cat=False,
            handle_feat_dynamic_real=False,
            handle_feat_dynamic_cat=True,
            hyperparameters=hyperparameters,
            process_hyperparameters=process_hyperparameters,
        )

        self._transformation += PowerTransformation()
        self._transformation += Detrend()

    def tune_config(self, prediction_length) -> Dict[str, Any]:
        """Select parameters in the pre-defined hyperparameter space.

        Returns:
            Selected parameters.
        """
        context_length = self._get_context_length(prediction_length)

        return {
            "method": self._auto_select("method"),
            "context_length": self._randint("context_length", context_length),
        }

    def build_estimator(
        self, *, freq: str, prediction_length: int, params: Dict[str, Any], **kwargs
    ) -> AAITimeSeriesEstimator:
        """Build an estimator from the underlying model using selected parameters.

        Args:
            freq: Frequency of the time series used.
            prediction_length: Length of the prediction that will be forecasted.
            params: Selected parameters from the hyperparameter space.
            kwargs: Ignored arguments.

        Returns:
            Built estimator.
        """
        return self._create_estimator(
            estimator=TreeEstimator(
                freq=freq,
                prediction_length=prediction_length,
                context_length=params.get("context_length", prediction_length),
                use_feat_dynamic_cat=self.use_feat_dynamic_cat,
                use_feat_dynamic_real=self.use_feat_dynamic_real,
                use_feat_static_real=self.use_feat_static_real,
                method=params["method"],
                quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
            ),
            additional_transformation=MultiDeseasonalizing(),  # FIXME try to remove from here
        )
