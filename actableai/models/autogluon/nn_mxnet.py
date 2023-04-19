from actableai.models.autogluon.base import BaseParams, Model
from actableai.parameters.boolean import BooleanSpace
from actableai.parameters.numeric import IntegerRangeSpace, FloatRangeSpace
from actableai.parameters.options import OptionsSpace
from actableai.parameters.parameters import Parameters
from .nn_base import get_parameters


class NNMXNetParams(BaseParams):
    """Parameter class for NN Model."""

    supported_problem_types = ["regression", "binary", "multiclass"]
    _autogluon_name = "NN_MXNET"
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
        # TODO: Check if any additional parameters should be added, such as
        # 'layers', 'loss_function', 'lr_scheduler', and parameters related to
        # 'lr_scheduler'
        parameters += [
            # MXNet-specific hyper-parameters:
            # TODO: Include settings for 'layers' by specifying each layer
            # individually. Need to add way to handle these when passing to AutoGluon
            # IntegerRangeSpace(
            #     name="hidden_layer_1_size",
            #     display_name="Hidden Layer 1 Size",
            #     description="Dimension of first layer.",
            #     default=40,
            #     min=1,
            #     # TODO check constraints
            # ),
            # IntegerRangeSpace(
            #     name="hidden_layer_2_size",
            #     display_name="Hidden Layer 2 Size",
            #     description="Dimension of second layer.",
            #     default=40,
            #     min=0,
            #     # TODO check constraints
            # ),
            # IntegerRangeSpace(
            #     name="hidden_layer_3_size",
            #     display_name="Hidden Layer 3 Size",
            #     description="Dimension of third layer.",
            #     default=0,
            #     min=0,
            #     # TODO check constraints
            # ),
            IntegerRangeSpace(
                name="max_layer_width",
                display_name="Maximum Layer Width",
                description="Maximum number of hidden units in network layer.",
                default=2056,
                min=1,  # Must be greater than 0
                hidden=True,
                # TODO check constraints (maximum)
            ),
            OptionsSpace[int](
                name="batch_size",
                display_name="Batch Size",
                description="Batch size used for training.",
                default=[512],
                options={
                    32: {"display_name": "32", "value": 32},
                    64: {"display_name": "64", "value": 64},
                    128: {"display_name": "128", "value": 128},
                    256: {"display_name": "256", "value": 256},
                    512: {"display_name": "512", "value": 512},
                    1024: {"display_name": "1024", "value": 1024},
                    2048: {"display_name": "2048", "value": 2048},
                },
                # TODO: check if use integer space; AutoGluon uses categories
                # TODO: AutoGluon uses 2056 instead of 2048 for default
                # value...check if need to update
            ),
            BooleanSpace(
                name="use_batchnorm",
                display_name="Batch Normalization",
                description="Whether to use batch normalization.",
                default="true",
            ),
            OptionsSpace[int](
                name="clip_gradient",
                display_name="Gradient Clipping Threshold",
                description="Gradient clipping threshold.",
                default=[100],
                is_multi=True,
                options={
                    1: {"display_name": "1", "value": 1},
                    10: {"display_name": "10", "value": 10},
                    100: {"display_name": "100", "value": 100},
                    1000: {"display_name": "1000", "value": 1000},
                },
                # TODO: check if use integer parameter/space; AutoGluon uses categories
            ),
            OptionsSpace[str](
                name="network_type",
                display_name="Network Type",
                description="The type of neural net used to produce predictions.",
                default=["widedeep"],
                is_multi=True,
                options={
                    "widedeep": {"display_name": "Wide-deep", "value": "widedeep"},
                    "feedforward": {
                        "display_name": "Feed-forward",
                        "value": "feedforward",
                    },
                },
                # TODO: check if add more loss functions; see https://pytorch.org/docs/stable/nn.html#loss-functions
            ),
            FloatRangeSpace(
                name="momentum",
                display_name="Momentum for SGD Optimizer",
                description="Momentum for SGD Optimizer.",
                default=0.9,
                min=0.01,
                # TODO check constraints
            ),
            FloatRangeSpace(
                name="base_lr",
                display_name="Smallest Learning Rate",
                description="Smallest learning rate that can be used.",
                default=3e-5,
                min=0.00001,  # Must be greater than 0
                # TODO check constraints
            ),
            FloatRangeSpace(
                name="target_lr",
                display_name="Largest Learning Rate",
                description="Largest learning rate that can be used.",
                default=1.0,
                min=0.00001,  # Must be greater than 0
                # TODO check constraints
            ),
            FloatRangeSpace(
                name="lr_decay",
                display_name="Learning Rate Decay Step Factor",
                description="Step factor used to decay LR.",
                default=0.1,
                min=0.00001,  # Must be greater than 0
                max=1,
                # TODO check constraints
            ),
            IntegerRangeSpace(
                name="warmup_epochs",
                display_name="Warmup Epochs",
                description="Number of Epochs at beginning of training in which the learning rate is linearly ramped up.",
                default=10,
                min=2,  # Must be greater than 1
                # TODO check constraints
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
                    "softrelu": {
                        "display_name": "Soft Rectified Exponential Linear Unit (Soft ReLU)",
                        "value": "softrelu",
                    },
                    "tanh": {"display_name": "tanh", "value": "tanh"},
                    "softsign": {"display_name": "Softsign", "value": "softsign"},
                },
            ),
        ]

        return Parameters(
            name=Model.nn_mxnet,
            display_name="Tabular Neural Network (MXNet)",
            parameters=parameters,
        )
