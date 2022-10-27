from functools import lru_cache
from typing import Dict, Any

from gluonts.model.n_beats import NBEATSEnsembleEstimator
from gluonts.mx import Trainer
from mxnet import Context

from actableai.parameters.numeric import FloatRangeSpace, IntegerRangeSpace
from actableai.parameters.parameters import Parameters
from actableai.timeseries.models.estimator import AAITimeSeriesEstimator
from actableai.timeseries.models.params.base import BaseParams, Model


class NBEATSParams(BaseParams):
    """Parameter class for NBEATS Model."""

    @staticmethod
    @lru_cache(maxsize=None)
    def get_hyperparameters() -> Parameters:
        """Returns the hyperparameters space of the model.

        Returns:
            The hyperparameters space.
        """

        parameters = [
            FloatRangeSpace(
                name="context_length_ratio",
                display_name="Context Length Ratio",
                description="Number of steps to unroll the RNN for before computing predictions. The Context Length is computed by multiplying this ratio with the Prediction Length.",
                default=(1, 2),
                min=1,
                # TODO check constraints
            ),
            IntegerRangeSpace(
                name="epochs",
                display_name="Epochs",
                # TODO add description
                description="description_epochs_todo",
                default=(5, 20),
                min=1,
                # TODO check constraints
            ),
            FloatRangeSpace(
                name="learning_rate",
                display_name="Learning Rate",
                # TODO add description
                description="description_learning_rate_todo",
                default=(0.0001, 0.01),
                min=0.0001,
                # TODO check constraints
            ),
            FloatRangeSpace(
                name="l2",
                display_name="L2 Regularization",
                # TODO add description
                description="description_l2_todo",
                default=(1e-4, 0.01),
                min=0,
                # TODO check constraints
            ),
            IntegerRangeSpace(
                name="meta_bagging_size",
                display_name="Meta Bagging Size",
                # TODO add description
                description="description_meta_bagging_size_todo",
                default=10,
                # TODO check constraints
            ),
        ]

        return Parameters(
            name=Model.n_beats,
            display_name="NBEATS Estimator",
            parameters=parameters,
        )

    def __init__(
        self,
        hyperparameters: Dict = None,
        process_hyperparameters: bool = True,
    ):
        """NBEATSParams Constructor.

        Args:
            hyperparameters: Dictionary representing the hyperparameters space.
            process_hyperparameters: If True the hyperparameters will be validated and
                processed (deactivate if they have already been validated).
        """
        super().__init__(
            model_name="NBEATS",
            is_multivariate_model=False,
            has_estimator=True,
            handle_feat_static_real=False,
            handle_feat_static_cat=False,
            handle_feat_dynamic_real=False,
            handle_feat_dynamic_cat=False,
            hyperparameters=hyperparameters,
            process_hyperparameters=process_hyperparameters,
        )

    def tune_config(self, prediction_length) -> Dict[str, Any]:
        """Select parameters in the pre-defined hyperparameter space.

        Returns:
            Selected parameters.
        """
        context_length = self._get_context_length(prediction_length)

        return {
            "context_length": self._randint("context_length", context_length),
            "epochs": self._auto_select("epochs"),
            "learning_rate": self._auto_select("learning_rate"),
            "l2": self._auto_select("l2"),
            "meta_bagging_size": self._auto_select("meta_bagging_size"),
        }

    def build_estimator(
        self,
        *,
        ctx: Context,
        freq: str,
        prediction_length: int,
        params: Dict[str, Any],
        **kwargs
    ) -> AAITimeSeriesEstimator:
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
        return self._create_estimator(
            NBEATSEnsembleEstimator(
                freq=freq,
                prediction_length=prediction_length,
                meta_context_length=[params.get("context_length", prediction_length)],
                meta_loss_function=["MAPE"],
                meta_bagging_size=params["meta_bagging_size"],
                trainer=Trainer(
                    ctx=ctx,
                    epochs=params["epochs"],
                    learning_rate=params["learning_rate"],
                    weight_decay=params["l2"],
                    hybridize=False,
                ),
            )
        )
