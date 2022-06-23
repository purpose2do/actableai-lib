from typing import Dict, Any, Optional, Union, Tuple

from gluonts.model.n_beats import NBEATSEnsembleEstimator
from gluonts.mx import DistributionOutput, Trainer
from mxnet import Context

from actableai.timeseries.models.params.base import BaseParams


class NBEATSParams(BaseParams):
    """Parameter class for NBEATS Model."""

    def __init__(
        self,
        context_length: Union[Tuple[int, int], int] = (1, 100),
        epochs: Union[Tuple[int, int], int] = (1, 100),
        learning_rate: Union[Tuple[float, float], float] = (0.0001, 0.01),
        l2: Union[Tuple[float, float], float] = (1e-4, 0.01),
        meta_bagging_size: Union[Tuple[int, int], int] = 10,
    ):
        """NBEATSParams Constructor.

        Args:
            context_length: Number of time units that condition the predictions, if
                tuple it represents the minimum and maximum (excluded) value.
            epochs: Number of epochs, if tuple it represents the minimum and maximum
                (excluded) value.
            learning_rate: Learning rate parameter, if tuple it represents the minimum
                and maximum (excluded) value.
            l2: L2 regularization parameter, if tuple it represents the minimum and
                maximum (excluded) value.
            meta_bagging_size: Bagging size (number of model to create for the
                ensemble), if tuple it represents the minimum and the maximum (excluded)
                value.
        """
        super().__init__(
            model_name="NBEATS",
            is_multivariate_model=False,
            has_estimator=True,
            handle_feat_static_real=False,
            handle_feat_static_cat=False,
            handle_feat_dynamic_real=False,
            handle_feat_dynamic_cat=False,
        )

        self.context_length = context_length
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.meta_bagging_size = meta_bagging_size

    def tune_config(self) -> Dict[str, Any]:
        """Select parameters in the pre-defined hyperparameter space.

        Returns:
            Selected parameters.
        """
        return {
            "context_length": self._randint("context_length", self.context_length),
            "epochs": self._randint("epochs", self.epochs),
            "learning_rate": self._uniform("learning_rate", self.learning_rate),
            "l2": self._uniform("l2", self.l2),
            "meta_bagging_size": self._randint("meta_bagging_size", self.meta_bagging_size),
        }

    def build_estimator(
        self,
        *,
        ctx: Context,
        freq: str,
        prediction_length: int,
        params: Dict[str, Any],
        **kwargs
    ) -> NBEATSEnsembleEstimator:
        """Build an estimator from the underlying model using selected parameters.

        Args:
            ctx: mxnet context.
            freq: Frequency of the time series used.
            prediction_length: Length of the prediction that will be forecasted.
            params: Selected parameters from the hyperparameter space.
            kwargs: Ignored arguments.

        Returns:
            Built estimator.
        """
        return NBEATSEnsembleEstimator(
            freq=freq,
            prediction_length=prediction_length,
            meta_context_length=[params.get("context_length", self.context_length)],
            meta_loss_function=["MAPE"],
            meta_bagging_size=params.get("meta_bagging_size", self.meta_bagging_size),
            trainer=Trainer(
                ctx=ctx,
                epochs=params.get("epochs", self.epochs),
                learning_rate=params.get("learning_rate", self.learning_rate),
                weight_decay=params.get("l2", self.l2),
                hybridize=False,
            )
        )
