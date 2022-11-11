from functools import lru_cache
from typing import Dict, Any

from gluonts.model.r_forecast import RForecastPredictor

from actableai.parameters.options import OptionsSpace
from actableai.parameters.parameters import Parameters
from actableai.timeseries.models.params.base import BaseParams, Model
from actableai.timeseries.models.predictor import AAITimeSeriesPredictor


class RForecastParams(BaseParams):
    """Parameters class for RForecast Model."""

    @staticmethod
    @lru_cache(maxsize=None)
    def get_hyperparameters() -> Parameters:
        """Returns the hyperparameters space of the model.

        Returns:
            The hyperparameters space.
        """

        parameters = [
            OptionsSpace[str](
                name="method_name",
                display_name="Method Name",
                description="Forecasting method to use.",
                default=["arima", "ets"],
                options={
                    "tbats": {"display_name": "TBATS", "value": "tbats"},
                    "thetaf": {"display_name": "THETAF", "value": "thetaf"},
                    "stlar": {"display_name": "STLAR", "value": "stlar"},
                    "arima": {"display_name": "ARIMA", "value": "arima"},
                    "ets": {"display_name": "ETS", "value": "ets"},
                },
            ),
        ]

        return Parameters(
            name=Model.r_forecast,
            display_name="R Forecast Predictor",
            parameters=parameters,
        )

    def __init__(
        self,
        hyperparameters: Dict = None,
        process_hyperparameters: bool = True,
    ):
        """RForecastParams Constructor.

        Args:
            hyperparameters: Dictionary representing the hyperparameters space.
            process_hyperparameters: If True the hyperparameters will be validated and
                processed (deactivate if they have already been validated).
        """
        super().__init__(
            model_name="RForecast",
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
            "method_name": self._auto_select("method_name"),
        }

    def build_predictor(
        self, *, freq: str, prediction_length: int, params: Dict[str, Any], **kwargs
    ) -> AAITimeSeriesPredictor:
        """Build a predictor from the underlying model using selected parameters.

        Args:
            freq: Frequency of the time series used.
            prediction_length: Length of the prediction that will be forecasted.
            params: Selected parameters from the hyperparameter space.
            kwargs: Ignored arguments.

        Returns:
            Built predictor.
        """
        return self._create_predictor(
            RForecastPredictor(
                freq=freq,
                prediction_length=prediction_length,
                method_name=params["method_name"],
            )
        )
