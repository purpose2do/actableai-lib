from actableai.models.autogluon.base import BaseParams, Model
from actableai.parameters.numeric import (
    FloatRangeSpace,
    IntegerRangeSpace,
    FloatParameter,
)
from actableai.parameters.options import OptionsParameter
from actableai.parameters.parameters import Parameters


class FastAINNParams(BaseParams):
    """Parameter class for fastai Model."""

    supported_problem_types = ["regression", "quantile", "binary", "multiclass"]
    _autogluon_name = "FASTAI"
    explain_samples_supported = False

    @classmethod
    def _get_hyperparameters(cls, *, problem_type: str, **kwargs) -> Parameters:
        """Returns the hyperparameters space of the model.

         Args:
            problem_type: Defines the type of the problem (e.g. regression,
                binary classification, etc.). See
                cls.supported_problem_types
                for list of accepted strings

        Returns:
            The hyperparameters space.
        """

        # See
        # https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/fastainn/hyperparameters/parameters.py,
        # https://auto.gluon.ai/0.6.0/_modules/autogluon/tabular/models/fastainn/tabular_nn_fastai.html,
        # https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/fastainn/tabular_nn_fastai.py
        # For AutoGluon hyperparameter search spaces, see
        # https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/fastainn/hyperparameters/searchspaces.py
        # https://fastai1.fast.ai/tabular.models.html
        # NOTE: There seems to be some discrepancies between variable names used
        # by AutoGluon and names used by fastai; parameters below are based on
        # names used in AutoGluon. May need to be eventually updated.
        # TODO: Check if any additional parameters should be added, such as
        # layers, bs, epochs, smoothing
        parameters = [
            FloatRangeSpace(
                name="lr",
                display_name="Learning Rate",
                description="Maximum learning rate for '1cycle' policy.",
                default=1e-2,
                min=0.0001,
                _log=True,
                # TODO check constraints
            ),
            FloatRangeSpace(
                name="emb_drop",
                display_name="Embedding Layers Dropout Probability",
                description="Dropout probability of embedding layers.",
                default=0.1,
                min=0,
                max=1.00001,
                # TODO check constraints, range
            ),
            FloatRangeSpace(
                name="ps",
                display_name="Linear Layers Dropout Probability",
                description="Dropout probability of linear layers.",
                default=0.1,
                min=0,
                max=1.00001,
                # TODO check constraints, range
            ),
            FloatRangeSpace(
                name="early.stopping.min_delta",
                display_name="Early Stopping Delta",
                description="Minimum delta between the last monitor value and the best monitor value.",
                default=0.0001,
                min=0,
                max=1.00001,
                _log=True,
                # TODO check constraints, range
            ),
            IntegerRangeSpace(
                name="early.stopping.patience",
                display_name="Early Stopping Patience",
                description="Number of epochs to wait when training has not improved model.",
                default=20,
                min=1,
                # TODO check constraints, range
            ),
            OptionsParameter[str](
                name="bs",
                display_name="Batch Size",
                description="The batch size.",
                default="auto",
                hidden=True,
                is_multi=False,
                options={
                    "auto": {"display_name": "Auto", "value": "auto"},
                },
            ),
            OptionsParameter[str](
                name="Epochs",
                display_name="Epochs",
                description="Maximum number of epochs.",
                default="auto",
                hidden=True,
                is_multi=False,
                options={
                    "auto": {"display_name": "Auto", "value": "auto"},
                },
            ),
            FloatParameter(
                name="smoothing",
                display_name="Smoothing",
                description="Smoothing. If greater than 0, then use LabelSmoothingCrossEntropy loss function for binary/multi-class classification; otherwise use default loss function for this type of problem.",
                default=0.0,
                hidden=True,
                # TODO check constraints
            ),
        ]

        if problem_type == "quantile":
            parameters += [
                FloatRangeSpace(
                    name="alpha",
                    display_name="Alpha",
                    description="Residual threshold parameter in Huber Pinball Loss.",
                    default=0.01,
                    min=0,
                    # TODO check constraints (e.g. can this be negative?), range
                    # TODO: Use categorical as used by AutoGluon?
                ),
            ]

        return Parameters(
            name=Model.nn_fastainn,
            display_name="fastai",
            parameters=parameters,
        )
