from typing import List

from actableai.parameters.base import BaseParameter
from actableai.parameters.boolean import BooleanSpace
from actableai.parameters.numeric import FloatRangeSpace, IntegerRangeSpace
from actableai.parameters.options import OptionsSpace


def get_parameters() -> List[BaseParameter]:
    """
    Define parameters that are common for all variants of NN models (PyTorch and
    MXNet)

    Args:
        problem_type: Defines the type of the problem (e.g. regression,
            binary classification, etc.). See
            nn_torch.supported_problem_types and
            nn_mxnet.supported_problem_types for list of accepted strings

    Returns:
        parameters: list containing the defined parameters
    """

    # See
    # https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/tabular_nn/hyperparameters/parameters.py
    # For AutoGluon hyperparameter search spaces, see
    # https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/tabular_nn/hyperparameters/searchspaces.py
    # TODO: Check if any additional parameters should be added
    parameters = [
        IntegerRangeSpace(
            name="num_epochs",
            display_name="Maximum Number of Epochs",
            description="The maximum number of epochs (passed over full dataset) for training the neural network.",
            default=500,
            min=1,
            hidden=True,
            # TODO check constraints (maximum)
        ),
        IntegerRangeSpace(
            name="epochs_wo_improve",
            display_name="Termination Epochs",
            description="Terminate training if validation performance has not improved in the last N epochs, where N is the specified value.",
            default=20,
            min=1,
            hidden=True,
            # TODO check constraints (maximum)
            # NOTE: This might be changed by AutoGluon to use smarter logic
        ),
        FloatRangeSpace(
            name="learning_rate",
            display_name="Learning Rate",
            description="Learning rate used during training.",
            default=3e-4,
            min=0.0001,  # Must be greater than 0
            # TODO check constraints (maximum)
            # NOTE: AutoGluon sets log=True (log-scale)
        ),
        FloatRangeSpace(
            name="embedding_size_factor",
            display_name="Embedding Size Factor",
            description="Scaling factor to adjust size of embedding layers.",
            default=1.0,
            min=0.01,
            max=100.0001,
            _log=True,
            # NOTE: Log-scale
        ),
        OptionsSpace[float](
            name="dropout_prob",
            display_name="Dropout Probability",
            description="Dropout probability, where a value of 0 turns off dropout.",
            default=[0.1],
            is_multi=True,
            options={
                0.0: {"display_name": "0.0", "value": 0.0},
                0.1: {"display_name": "0.1", "value": 0.1},
                0.2: {"display_name": "0.2", "value": 0.2},
                0.3: {"display_name": "0.3", "value": 0.3},
                0.4: {"display_name": "0.4", "value": 0.4},
                0.5: {"display_name": "0.5", "value": 0.5},
            },
            # TODO: Check if use options or FloatRangeSpace; AutoGluon uses categories
        ),
        OptionsSpace[int](
            name="proc.embed_min_categories",
            display_name="Minimum Levels for Embedding",
            description="Apply embedding layer to categorical features with at least this many levels. Features with fewer levels are one-hot encoded. Choose big value to avoid use of Embedding layers.",
            default=[4],
            is_multi=True,
            options={
                3: {"display_name": "3", "value": 3},
                4: {"display_name": "4", "value": 4},
                10: {"display_name": "10", "value": 10},
                100: {"display_name": "100", "value": 100},
                1000: {"display_name": "1000", "value": 1000},
            },
        ),
        OptionsSpace[str](
            name="proc.impute_strategy",
            display_name="Method to Impute Missing Values",
            description="How to impute missing numeric values.",
            default=["median"],
            is_multi=True,
            options={
                "median": {"display_name": "Median", "value": "median"},
                "mean": {"display_name": "Mean", "value": "mean"},
                "most_frequent": {
                    "display_name": "Most Frequent",
                    "value": "most_frequent",
                },
            },
        ),
        OptionsSpace[int](
            name="proc.max_category_levels",
            display_name="Maximum Levels per Categorical Feature",
            description="Maximum number of allowed levels per categorical feature.",
            default=[100],
            is_multi=True,
            options={
                10: {"display_name": "10", "value": 10},
                20: {"display_name": "20", "value": 20},
                100: {"display_name": "100", "value": 100},
                200: {"display_name": "200", "value": 200},
                300: {"display_name": "300", "value": 300},
                400: {"display_name": "400", "value": 400},
                500: {"display_name": "500", "value": 500},
                1000: {"display_name": "1000", "value": 1000},
                10000: {"display_name": "10000", "value": 10000},
            },
        ),
        OptionsSpace[float](
            name="proc.skew_threshold",
            display_name="Skewness Threshold",
            description="Numerical features whose absolute skewness is greater than this receive special power-transform preprocessing. Choose big value to avoid using power-transforms.",
            default=[0.99],
            is_multi=True,
            options={
                0.99: {"display_name": "0.99", "value": 0.99},
                0.2: {"display_name": "0.2", "value": 0.2},
                0.3: {"display_name": "0.3", "value": 0.3},
                0.5: {"display_name": "0.5", "value": 0.5},
                0.8: {"display_name": "0.8", "value": 0.8},
                0.9: {"display_name": "0.9", "value": 0.9},
                0.999: {"display_name": "0.999", "value": 0.999},
                1.0: {"display_name": "1.0", "value": 1.0},
                10.0: {"display_name": "10.0", "value": 10.0},
                100.0: {"display_name": "100.0", "value": 100.0},
            },
        ),
        OptionsSpace[str](
            name="optimizer",
            display_name="Optimizer",
            description="The optimizer to use for training.",
            default=["adam"],
            is_multi=True,
            options={
                "adam": {"display_name": "Adam", "value": "adam"},
                "sgd": {
                    "display_name": "Stochastic Gradient Descent (SGD)",
                    "value": "sgd",
                },
            },
        ),
        BooleanSpace(
            name="use_ngram_features",
            display_name="Use n-gram Features?",
            description="Whether to use n-gram features. If False, will drop automatically generated n-gram features from language features. This results in worse model quality but far faster inference and training times.",
            default="false",
        ),
        # TODO: Text/NLP-based parameter; check if should be included, or if
        # should only be included if a parameter is set
    ]

    # Defined separately so that it can be adjusted according to the problem
    # type, if necessary
    parameters += [
        FloatRangeSpace(
            name="weight_decay",
            display_name="Weight Decay",
            description="Weight decay regularizer.",
            default=1e-6,
            min=0,
            _log=True,
            # TODO check constraints, range
            # NOTE: AutoGluon sets log=True (log-scale)
        ),
    ]

    return parameters
