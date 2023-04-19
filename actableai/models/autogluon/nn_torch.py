from actableai.models.autogluon.base import BaseParams, Model
from actableai.parameters.boolean import BooleanSpace
from actableai.parameters.numeric import IntegerRangeSpace, FloatRangeSpace
from actableai.parameters.options import OptionsSpace
from actableai.parameters.parameters import Parameters
from .nn_base import get_parameters


# TODO: Uncomment package import if enable 'loss_function' option
# from torch import nn


class NNTorchParams(BaseParams):
    """Parameter class for NN Model."""

    # NOTE: 'regression' supported but disabled since it can be memory-intensive
    supported_problem_types = ["quantile", "binary", "multiclass"]
    _autogluon_name = "NN_TORCH"
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
        # Get parameters common to all variants:
        parameters = get_parameters()

        # See
        # https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/tabular_nn/hyperparameters/parameters.py
        # For AutoGluon hyperparameter search spaces, see
        # https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/tabular_nn/hyperparameters/searchspaces.py
        # TODO: Check if any additional parameters should be added
        parameters += [
            # Pytorch-specific hyper-parameters:
            IntegerRangeSpace(
                name="num_layers",
                display_name="Number of Layers",
                description="Number of layers in the network.",
                default=4,
                min=2,
                max=6,  # Up to 5 supported
                # TODO: Check if use options/categories as used by AutoGluon
            ),
            OptionsSpace[int](
                name="hidden_size",
                display_name="Number of Hidden Units per Layer",
                description="Number of hidden units in each layer.",
                default=[128],
                options={
                    128: {"display_name": "128", "value": 128},
                    256: {"display_name": "256", "value": 256},
                    512: {"display_name": "512", "value": 512},
                },
            ),
            IntegerRangeSpace(
                name="max_batch_size",
                display_name="Maximum Batch Size",
                description="Maximum batch size, actual batch size may be slightly smaller.",
                default=512,
                min=1,
                max=1025,
                # TODO: Check constraints, range
            ),
            BooleanSpace(
                name="use_batchnorm",
                display_name="Batch Normalization",
                description="Whether to use batch normalization.",
                default="false",
            ),
            OptionsSpace[str](
                name="loss_function",
                display_name="Loss Function",
                description="The loss function minimized during training.",
                default=["auto"],
                hidden=True,
                options={
                    "auto": {"display_name": "Auto", "value": "auto"},
                    # "nn.L1Loss()": {
                    #     "display_name": "Mean Absolute Error (MAE) (L1)",
                    #     "value": nn.L1Loss(),
                    # },
                    # "nn.MSELoss()": {
                    #     "display_name": "Mean Squared Error (MSE) (L2)",
                    #     "value": nn.MSELoss(),
                    # },
                    # "nn.HuberLoss()": {
                    #     "display_name": "Huber Loss",
                    #     "value": nn.HuberLoss(),
                    # },
                },
                # TODO: check if add more loss functions; see
                # https://pytorch.org/docs/stable/nn.html#loss-functions
                # TODO: Check values (use function directly or string?)
            ),
        ]

        # Parameter defined separately so that it can be adjusted according to the
        # problem type, if necessary
        parameters += [
            OptionsSpace[str](
                name="activation",
                display_name="Activation Function",
                description="Activation function to use in the model.",
                default=["relu"],
                is_multi=True,
                options={
                    "relu": {
                        "display_name": "Rectified Linear Unit (ReLU)",
                        "value": "relu",
                    },
                    "elu": {
                        "display_name": "Exponential Linear Unit (ELU)",
                        "value": "elu",
                    },
                    "tanh": {"display_name": "tanh", "value": "tanh"},
                },
            ),
        ]

        # Add some more options for quantile regression
        if problem_type == "quantile":
            parameters += [
                FloatRangeSpace(
                    name="gamma",
                    display_name="Gamma",
                    description="Margin loss weight which helps to ensure non-crossing quantile estimates.",
                    default=5,
                    min=0.1,
                    max=10.00001,
                ),
                OptionsSpace[float](
                    name="alpha",
                    display_name="Alpha",
                    description="Smoothing when using the Huber Pinball loss.",
                    default=[0.01],
                    is_multi=True,
                    options={
                        0.001: {"display_name": "0.001", "value": 0.001},
                        0.01: {"display_name": "0.01", "value": 0.01},
                        0.1: {"display_name": "0.1", "value": 0.1},
                        1.0: {"display_name": "1.0", "value": 1.0},
                    },
                ),
                # TODO: Check if use Float range or categorical (AutoGluon uses
                # categorical)
            ]

        return Parameters(
            name=Model.nn_torch,
            display_name="Tabular Neural Network (PyTorch)",
            parameters=parameters,
        )
