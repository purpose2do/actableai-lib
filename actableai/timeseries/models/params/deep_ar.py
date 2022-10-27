from functools import lru_cache
from typing import Dict, Any

from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.distribution import DistributionOutput
from gluonts.mx.trainer import Trainer
from mxnet.context import Context

from actableai.parameters.boolean import BooleanSpace
from actableai.parameters.numeric import IntegerRangeSpace, FloatRangeSpace
from actableai.parameters.parameters import Parameters
from actableai.timeseries.models.estimator import AAITimeSeriesEstimator
from actableai.timeseries.models.params.base import BaseParams, Model


class DeepARParams(BaseParams):
    """Parameter class for Deep AR Model."""

    @staticmethod
    @lru_cache(maxsize=None)
    def get_hyperparameters() -> Parameters:
        """Returns the hyperparameters space of the model.

        Returns:
            The hyperparameters space.
        """

        parameters = [
            IntegerRangeSpace(
                name="epochs",
                display_name="Epochs",
                # TODO add description
                description="description_epochs_todo",
                default=(1, 20),
                min=1,
                # TODO check constraints
            ),
            IntegerRangeSpace(
                name="num_cells",
                display_name="Number of Cells",
                # TODO add description
                description="description_num_cells_todo",
                default=(1, 10),
                min=1,
                # TODO check constraints
            ),
            IntegerRangeSpace(
                name="num_layers",
                display_name="Number of Layers",
                # TODO add description
                description="description_num_layers_todo",
                default=(1, 3),
                min=1,
                # TODO check constraints
            ),
            FloatRangeSpace(
                name="dropout_rate",
                display_name="Dropout Rate",
                # TODO add description
                description="description_dropout_rate_todo",
                default=(0, 0.5),
                min=0,
                max=1,
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
                # TODO add description
                description="description_l2_todo",
                default=(1e-8, 0.01),
                min=0,
                # TODO check constraints
            ),
            BooleanSpace(
                name="scaling",
                display_name="Scaling",
                default="true",
                hidden=True,
            ),
            BooleanSpace(
                name="impute_missing_values",
                display_name="Impute Missing Values",
                default="true",
                hidden=True,
            ),
        ]

        return Parameters(
            name=Model.deep_ar,
            display_name="Deep AR Estimator",
            parameters=parameters,
        )

    def __init__(
        self, hyperparameters: Dict = None, process_hyperparameters: bool = True
    ):
        """DeepARParams Constructor.

        Args:
            hyperparameters: Dictionary representing the hyperparameters space.
            process_hyperparameters: If True the hyperparameters will be validated and
                processed (deactivate if they have already been validated).
        """
        super().__init__(
            model_name="DeepAR",
            is_multivariate_model=False,
            has_estimator=True,
            handle_feat_static_real=True,
            handle_feat_static_cat=False,
            handle_feat_dynamic_real=True,
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
            "dropout_rate": self._auto_select("dropout_rate"),
            "learning_rate": self._auto_select("learning_rate"),
            "context_length": self._randint("context_length", context_length),
            "l2": self._auto_select("l2"),
            "scaling": self._auto_select("scaling"),
            "impute_missing_values": self._auto_select("impute_missing_values"),
        }

    def build_estimator(
        self,
        *,
        ctx: Context,
        freq: str,
        prediction_length: int,
        distr_output: DistributionOutput,
        params: Dict[str, Any],
        **kwargs,
    ) -> AAITimeSeriesEstimator:
        """Build an estimator from the underlying model using selected parameters.

        Args:
            ctx: mxnet context.
            freq: Frequency of the time series used.
            prediction_length: Length of the prediction that will be forecasted
            distr_output: Distribution output to use.
            params: Selected parameters from the hyperparameter space.
            kwargs: Ignored arguments.

        Returns:
            Built estimator.
        """
        return self._create_estimator(
            DeepAREstimator(
                freq=freq,
                prediction_length=prediction_length,
                cell_type="lstm",
                use_feat_dynamic_real=self.use_feat_dynamic_real,
                use_feat_static_cat=self.use_feat_static_cat,
                use_feat_static_real=self.use_feat_static_real,
                cardinality=None,
                scaling=params["scaling"],
                dropout_rate=params["dropout_rate"],
                num_layers=params["num_layers"],
                num_cells=params["num_cells"],
                context_length=params.get("context_length", prediction_length),
                distr_output=distr_output,
                impute_missing_values=params["impute_missing_values"],
                trainer=Trainer(
                    ctx=ctx,
                    epochs=params["epochs"],
                    learning_rate=params["learning_rate"],
                    hybridize=False,
                    weight_decay=params["l2"],
                ),
            )
        )
