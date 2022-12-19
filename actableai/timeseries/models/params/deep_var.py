from __future__ import annotations

from functools import lru_cache
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from mxnet.context import Context
    from actableai.parameters.parameters import Parameters
    from actableai.timeseries.models.estimator import AAITimeSeriesEstimator

from actableai.timeseries.models.params.base import BaseParams


class DeepVARParams(BaseParams):
    """Parameter class for Deep VAR Model."""

    @staticmethod
    @lru_cache(maxsize=None)
    def get_hyperparameters() -> Parameters:
        """Returns the hyperparameters space of the model.

        Returns:
            The hyperparameters space.
        """
        from actableai.parameters.boolean import BooleanSpace
        from actableai.parameters.numeric import IntegerRangeSpace, FloatRangeSpace
        from actableai.parameters.options import OptionsSpace
        from actableai.parameters.parameters import Parameters
        from actableai.timeseries.models.params.base import Model

        parameters = [
            IntegerRangeSpace(
                name="epochs",
                display_name="Epochs",
                description="Number of epochs used during training.",
                default=(5, 20),
                min=1,
                # TODO check constraints
            ),
            IntegerRangeSpace(
                name="num_cells",
                display_name="Number of Cells",
                description="Number of RNN cells for each layer.",
                default=(1, 20),
                min=1,
                # TODO check constraints
            ),
            IntegerRangeSpace(
                name="num_layers",
                display_name="Number of Layers",
                description="Number of RNN layers.",
                default=(1, 3),
                min=1,
                # TODO check constraints
            ),
            OptionsSpace[str](
                name="cell_type",
                display_name="Cell Type",
                description="Type of recurrent cells to use.",
                default=["lstm"],
                options={
                    "lstm": {"display_name": "LSTM", "value": "lstm"},
                    "gru": {"display_name": "GRU", "value": "gru"},
                },
                # TODO check available options
            ),
            FloatRangeSpace(
                name="dropout_rate",
                display_name="Dropout Rate",
                description="Dropout regularization rate used during training.",
                default=(0, 0.5),
                min=0,
                max=1,
                # TODO check constraints
            ),
            IntegerRangeSpace(
                name="rank",
                display_name="Rank",
                description="Rank for the LowrankMultivariateGaussianOutput.",
                default=5,
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
                name="learning_rate",
                display_name="Learning Rate",
                description="Learning rate used during training.",
                default=(0.0001, 0.01),
                min=0.0001,
                # TODO check constraints
            ),
            FloatRangeSpace(
                name="l2",
                display_name="L2 Regularization",
                description="L2 regularization used during training.",
                default=(1e-4, 0.01),
                min=0,
                # TODO check constraints
            ),
            BooleanSpace(
                name="scaling",
                display_name="Scaling",
                description="Whether to automatically scale the target values.",
                default="true",
                hidden=True,
            ),
        ]

        return Parameters(
            name=Model.deep_var,
            display_name="Deep VAR Estimator",
            parameters=parameters,
        )

    def __init__(
        self,
        hyperparameters: Dict = None,
        process_hyperparameters: bool = True,
    ):
        """DeepVARParams Constructor.

        Args:
            hyperparameters: Dictionary representing the hyperparameters space.
            process_hyperparameters: If True the hyperparameters will be validated and
                processed (deactivate if they have already been validated).
        """
        super().__init__(
            model_name="DeepVAR",
            is_multivariate_model=True,
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
            "epochs": self._auto_select("epochs"),
            "num_cells": self._auto_select("num_cells"),
            "num_layers": self._auto_select("num_layers"),
            "cell_type": self._auto_select("cell_type"),
            "dropout_rate": self._auto_select("dropout_rate"),
            "rank": self._auto_select("rank"),
            "context_length": self._randint("context_length", context_length),
            "learning_rate": self._auto_select("learning_rate"),
            "l2": self._auto_select("l2"),
            "scaling": self._auto_select("scaling"),
        }

    def build_estimator(
        self,
        *,
        ctx: Context,
        freq: str,
        prediction_length: int,
        target_dim: int,
        params: Dict[str, Any],
        **kwargs,
    ) -> AAITimeSeriesEstimator:
        """Build an estimator from the underlying model using selected parameters.

        Args:
            ctx: mxnet context.
            freq: Frequency of the time series used.
            prediction_length: Length of the prediction that will be forecasted.
            target_dim: Target dimension (number of columns to predict).
            params: Selected parameters from the hyperparameter space.
            kwargs: Ignored arguments.

        Returns:
            Built estimator.
        """
        from gluonts.model.deepvar import DeepVAREstimator
        from gluonts.mx.trainer import Trainer

        return self._create_estimator(
            DeepVAREstimator(
                freq=freq,
                prediction_length=prediction_length,
                cell_type=params["cell_type"],
                dropout_rate=params["dropout_rate"],
                num_layers=params["num_layers"],
                num_cells=params["num_cells"],
                context_length=params.get("context_length", prediction_length),
                num_parallel_samples=100,
                rank=params["rank"],
                scaling=params["scaling"],
                pick_incomplete=True,
                conditioning_length=100,
                use_marginal_transformation=False,
                target_dim=target_dim,
                trainer=Trainer(
                    ctx=ctx,
                    epochs=params["epochs"],
                    learning_rate=params["learning_rate"],
                    weight_decay=params["l2"],
                    hybridize=True,
                ),
            )
        )
