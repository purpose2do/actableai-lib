from functools import lru_cache
from typing import Dict, Any

from gluonts.model.prophet import ProphetPredictor

from actableai.parameters.options import OptionsSpace
from actableai.parameters.parameters import Parameters
from actableai.timeseries.models.params.base import BaseParams, Model
from actableai.timeseries.models.predictor import AAITimeSeriesPredictor


class ProphetParams(BaseParams):
    """Parameter class for Prophet Model."""

    @staticmethod
    @lru_cache(maxsize=None)
    def get_hyperparameters() -> Parameters:
        """Returns the hyperparameters space of the model.

        Returns:
            The hyperparameters space.
        """

        parameters = [
            OptionsSpace[str](
                name="growth",
                display_name="Growth",
                # TODO add description
                description="description_growth",
                default=["linear"],
                options={
                    "linear": {"display_name": "Linear", "value": "linear"},
                    "logistic": {"display_name": "Logistic", "value": "logistic"},
                    "flat": {"display_name": "Flat", "value": "flat"},
                },
                # TODO check available options
            ),
            OptionsSpace[str](
                name="seasonality_mode",
                display_name="Seasonality Mode",
                # TODO add description
                description="description_seasonality_mode",
                default=["additive", "multiplicative"],
                options={
                    "additive": {"display_name": "Additive", "value": "additive"},
                    "multiplicative": {
                        "display_name": "Multiplicative",
                        "value": "multiplicative",
                    },
                },
                # TODO check available options
            ),
        ]

        return Parameters(
            name=Model.prophet,
            display_name="Prophet Predictor",
            parameters=parameters,
        )

    def __init__(
        self,
        hyperparameters: Dict = None,
        process_hyperparameters: bool = True,
    ):
        """ProphetParams Constructor.

        Args:
            hyperparameters: Dictionary representing the hyperparameters space.
            process_hyperparameters: If True the hyperparameters will be validated and
                processed (deactivate if they have already been validated).
        """
        super().__init__(
            model_name="Prophet",
            is_multivariate_model=False,
            has_estimator=False,
            handle_feat_static_real=True,
            handle_feat_static_cat=True,
            handle_feat_dynamic_real=True,
            handle_feat_dynamic_cat=True,
            hyperparameters=hyperparameters,
            process_hyperparameters=process_hyperparameters,
        )

    def tune_config(self, prediction_length) -> Dict[str, Any]:
        """Select parameters in the pre-defined hyperparameter space.

        Returns:
            Selected parameters.
        """
        return {
            "growth": self._auto_select("growth"),
            "seasonality_mode": self._auto_select("seasonality_mode"),
        }

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
            ProphetPredictor(
                prediction_length=prediction_length,
                prophet_params={
                    "growth": params["growth"],
                    "seasonality_mode": params["seasonality_mode"],
                },
            )
        )
