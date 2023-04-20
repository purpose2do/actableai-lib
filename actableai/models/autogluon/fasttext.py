from actableai.models.autogluon.base import BaseParams, Model
from actableai.parameters.parameters import Parameters

from actableai.parameters.numeric import FloatRangeSpace, IntegerRangeSpace
from actableai.parameters.options import OptionsSpace


class FASTTEXTParams(BaseParams):
    """Parameter class for FASTTEXT Model."""

    # See https://auto.gluon.ai/0.5.2/_modules/autogluon/tabular/models/fasttext/fasttext_model.html

    # TODO: Check supported problem types
    supported_problem_types = ["binary", "multiclass"]
    _autogluon_name = "FASTTEXT"
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
        #   https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/fasttext/hyperparameters/parameters.py,
        #   https://fasttext.cc/docs/en/python-module.html#train_supervised-parameters
        parameters = [
            FloatRangeSpace(
                name="lr",
                display_name="Learning Rate",
                description="Learning rate used during training.",
                default=0.1,
                min=0.0001,
                # TODO check constraints
            ),
            IntegerRangeSpace(
                name="dim",
                display_name="Word Vector Size",
                description="Size of word vectors.",
                default=100,
                min=1,
                # TODO check constraints
            ),
            IntegerRangeSpace(
                name="ws",
                display_name="Context Window Size",
                description="Size of the context window.",
                default=5,
                min=1,
                # TODO check constraints
            ),
            IntegerRangeSpace(
                name="num_epochs",
                display_name="Number of Epochs",
                description="The number of epochs for training the model.",
                default=50,
                min=1,
                # TODO check constraints (maximum)
            ),
            IntegerRangeSpace(
                name="minCount",
                display_name="Minimal Word Occurrence Number",
                description="Minimal number of word occurrences.",
                default=1,
                min=1,
                # TODO check constraints (maximum)
            ),
            IntegerRangeSpace(
                name="minCount",
                display_name="Minimal label Occurrence Number",
                description="Minimal number of label occurrences.",
                default=1,
                min=1,
                # TODO check constraints (maximum)
            ),
            IntegerRangeSpace(
                name="minn",
                display_name="Minimal Character Ngram Length",
                description="Minimum length of character Ngram.",
                default=2,
                min=1,
                # TODO check constraints (maximum)
            ),
            IntegerRangeSpace(
                name="maxn",
                display_name="Maximal Character Ngram Length",
                description="Maximal length of character Ngram.",
                default=2,
                min=1,
                # TODO check constraints (maximum)
            ),
            IntegerRangeSpace(
                name="neg",
                display_name="Number of Negatives Sampled",
                description="Number of negatives sampled.",
                default=5,
                min=1,
                # TODO check constraints (maximum)
            ),
            IntegerRangeSpace(
                name="wordNgrams",
                display_name="Maximal Word Ngram length",
                description="Maximal length of word Ngram.",
                default=3,
                min=1,
                # TODO check constraints (maximum)
            ),
            OptionsSpace[str](
                name="loss",
                display_name="Loss Function",
                description="Loss Function to use.",
                default="softmax",
                options={
                    "softmax": {"display_name": "Softmax", "value": "softmax"},
                    "ns": {"display_name": "ns", "value": "ns"},
                    "hs": {"display_name": "hs", "value": "hs"},
                    "ova": {"display_name": "ova", "value": "ova"},
                },
                # TODO: Check if use options or IntegerRangeSpace; AutoGluon uses categories
            ),
            IntegerRangeSpace(
                name="bucket",
                display_name="Number of Buckets",
                description="Number of buckets.",
                default=2000000,
                min=1,
                # TODO check constraints
            ),
            IntegerRangeSpace(
                name="lrUpdateRate",
                display_name="Learning Rate Update Rate",
                description="Change the rate of updates for the learning rate.",
                default=100,
                min=1,
                # TODO check constraints
            ),
            FloatRangeSpace(
                name="t",
                display_name="Sampling Threshold",
                description="sampling threshold.",
                default=0.0001,
                min=1,
                # TODO check constraints
            ),
        ]
        return Parameters(
            name=Model.fasttext,
            display_name="FastText",
            parameters=parameters,
        )
