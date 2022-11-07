from functools import lru_cache
from typing import Dict, Any

from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.mx.distribution import DistributionOutput
from gluonts.mx.trainer import Trainer
from mxnet.context import Context

from actableai.parameters.boolean import BooleanSpace
from actableai.parameters.numeric import IntegerRangeSpace, FloatRangeSpace
from actableai.parameters.parameters import Parameters
from actableai.timeseries.models.estimator import AAITimeSeriesEstimator
from actableai.timeseries.models.params.base import BaseParams, Model


class FeedForwardParams(BaseParams):
    """Parameter class for Feed Forward Model."""

    @staticmethod
    @lru_cache(maxsize=None)
    def get_hyperparameters() -> Parameters:
        """Returns the hyperparameters space of the model.

        Returns:
            The hyperparameters space.
        """

        parameters = [
            IntegerRangeSpace(
                name="hidden_layer_1_size",
                display_name="Hidden Layer 1 Size",
                description="Dimension of first layer.",
                default=40,
                min=1,
                # TODO check constraints
            ),
            IntegerRangeSpace(
                name="hidden_layer_2_size",
                display_name="Hidden Layer 2 Size",
                description="Dimension of second layer.",
                default=40,
                min=0,
                # TODO check constraints
            ),
            IntegerRangeSpace(
                name="hidden_layer_3_size",
                display_name="Hidden Layer 3 Size",
                description="Dimension of third layer.",
                default=0,
                min=0,
                # TODO check constraints
            ),
            IntegerRangeSpace(
                name="epochs",
                display_name="Epochs",
                description="Number of epochs used during training.",
                default=(1, 1000),
                min=1,
                # TODO check constraints
            ),
            FloatRangeSpace(
                name="learning_rate",
                display_name="Learning Rate",
                description="Learning rate used during training.",
                default=(0.001, 0.01),
                min=0.0001,
                # TODO check constraints
            ),
            FloatRangeSpace(
                name="context_length_ratio",
                display_name="Context Length Ratio",
                description="Number of steps to unroll the RNN for before computing predictions. The Context Length is computed by multiplying this ratio with the Prediction Length.",
                default=(1, 2),
                min=1,
                # TODO check constraints
            ),
            FloatRangeSpace(
                name="l2",
                display_name="L2 Regularization",
                description="L2 regularization used during training.",
                default=(1e-8, 0.01),
                min=0,
                # TODO check constraints
            ),
            BooleanSpace(
                name="mean_scaling",
                display_name="Mean Scaling",
                description="Scale the network input by the data mean and the network output by its inverse.",
                default="true",
                hidden=True,
            ),
        ]

        return Parameters(
            name=Model.feed_forward,
            display_name="Feed Forward Estimator",
            parameters=parameters,
        )

    def __init__(
        self,
        hyperparameters: Dict = None,
        process_hyperparameters: bool = True,
    ):
        """FeedForwardParams Constructor.

        Args:
            hyperparameters: Dictionary representing the hyperparameters space.
            process_hyperparameters: If True the hyperparameters will be validated and
                processed (deactivate if they have already been validated).
        """
        super().__init__(
            model_name="FeedForward",
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
            "hidden_layer_1_size": self._auto_select("hidden_layer_1_size"),
            "hidden_layer_2_size": self._auto_select("hidden_layer_2_size"),
            "hidden_layer_3_size": self._auto_select("hidden_layer_3_size"),
            "epochs": self._auto_select("epochs"),
            "learning_rate": self._auto_select("learning_rate"),
            "context_length": self._randint("context_length", context_length),
            "l2": self._auto_select("l2"),
            "mean_scaling": self._auto_select("mean_scaling"),
        }

    def build_estimator(
        self,
        *,
        ctx: Context,
        prediction_length: int,
        distr_output: DistributionOutput,
        params: Dict[str, Any],
        **kwargs,
    ) -> AAITimeSeriesEstimator:
        """Build an estimator from the underlying model using selected parameters.

        Args:
            ctx: mxnet context.
            prediction_length: Length of the prediction that will be forecasted.
            distr_output: Distribution output to use.
            params: Selected parameters from the hyperparameter space.
            kwargs: Ignored arguments.

        Returns:
            Built estimator.
        """
        hidden_layer_1_size = params["hidden_layer_1_size"]
        hidden_layer_2_size = params["hidden_layer_2_size"]
        hidden_layer_3_size = params["hidden_layer_3_size"]

        hidden_layer_size = []
        if hidden_layer_1_size is not None and hidden_layer_1_size > 0:
            hidden_layer_size.append(hidden_layer_1_size)
        if hidden_layer_2_size is not None and hidden_layer_2_size > 0:
            hidden_layer_size.append(hidden_layer_2_size)
        if hidden_layer_3_size is not None and hidden_layer_3_size > 0:
            hidden_layer_size.append(hidden_layer_3_size)

        return self._create_estimator(
            SimpleFeedForwardEstimator(
                prediction_length=prediction_length,
                num_hidden_dimensions=hidden_layer_size,
                context_length=params.get("context_length", prediction_length),
                mean_scaling=params["mean_scaling"],
                distr_output=distr_output,
                batch_normalization=False,
                trainer=Trainer(
                    ctx=ctx,
                    epochs=params["epochs"],
                    learning_rate=params["learning_rate"],
                    weight_decay=params["l2"],
                    hybridize=False,
                ),
            )
        )
