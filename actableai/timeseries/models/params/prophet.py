from typing import Union, Tuple, Dict, Any

from gluonts.model.prophet import ProphetPredictor

from actableai.timeseries.models.params.base import BaseParams
from actableai.timeseries.models.predictor import AAITimeSeriesPredictor


class ProphetParams(BaseParams):
    """Parameter class for Prophet Model."""

    def __init__(
        self,
        growth: Union[Tuple[str, ...], str] = "linear",
        seasonality_mode: Union[Tuple[str, ...], str] = ("additive", "multiplicative"),
    ):
        """ProphetParams Constructor.

        Args:
            growth: Specify trend ["linear", "logistic", "flat"], if tuple it represents
                the different values to choose from.
            seasonality_mode: Seasonality mode parameter ["additive", "multiplicative"],
                if tuple it represents the different values to choose from.
        """
        super().__init__(
            model_name="Prophet",
            is_multivariate_model=False,
            has_estimator=False,
            handle_feat_static_real=True,
            handle_feat_static_cat=True,
            handle_feat_dynamic_real=True,
            handle_feat_dynamic_cat=True,
        )

        self.growth = growth
        self.seasonality_mode = seasonality_mode

    def tune_config(self) -> Dict[str, Any]:
        """Select parameters in the pre-defined hyperparameter space.

        Returns:
            Selected parameters.
        """
        return {
            "growth": self._choice("growth", self.growth),
            "seasonality_mode": self._choice("seasonality_mode", self.seasonality_mode),
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
                    "growth": params.get("growth", self.growth),
                    "seasonality_mode": params.get(
                        "seasonality_mode", self.seasonality_mode
                    ),
                },
            )
        )
