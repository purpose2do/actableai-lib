from __future__ import annotations

from typing import TYPE_CHECKING, Union, Type

if TYPE_CHECKING:
    from autogluon.core.models import AbstractModel

from actableai.parameters.parameters import Parameters
from actableai.models.autogluon.base import BaseParams, Model

from actableai.parameters.numeric import (
    FloatParameter,
    FloatRangeSpace,
    IntegerParameter,
    IntegerRangeSpace,
)
from actableai.parameters.options import OptionsSpace


class TabPFNParams(BaseParams):
    """Parameter class for TabPFN Model."""

    # TODO: Check supported problem types
    supported_problem_types = ["binary", "multiclass"]
    _autogluon_name = "invalid"
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
        # https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/tab_transformer/hyperparameters/parameters.py
        # For AutoGluon search spaces, see
        #   https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/tab_transformer/hyperparameters/searchspaces.py
        parameters = [
            FloatRangeSpace(
                name="lr",
                display_name="Learning Rate",
                description="Learning rate used during training.",
                default=3.6e-3,
                min=0.0001,
                # TODO check constraints
                # AutoGluon Options: Real(5e-5, 5e-3)
            ),
            FloatRangeSpace(
                name="weight_decay",
                display_name="Weight Decay",
                description="Rate of linear weight decay for learning rate.",
                default=1e-6,
                min=1e-6,
                # TODO check constraints
                # AutoGluon Options: Real(1e-6, 5e-2)
            ),
            OptionsSpace[float](
                name="p_dropout",
                display_name="Dropout Probability",
                description="Dropout probability, where a value of 0 turns off dropout.",
                default=0.0,
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
                name="n_heads",
                display_name="Number of Attention Heads",
                description="Number of attention heads.",
                default=4,
                options={
                    2: {"display_name": "2", "value": 2},
                    4: {"display_name": "4", "value": 4},
                    8: {"display_name": "8", "value": 8},
                },
                # TODO: Check if use options or IntRangeSpace; AutoGluon uses categories
            ),
            OptionsSpace[int](
                name="hidden_dim",
                display_name="Number of Hidden Units per Layer",
                description="Number of hidden units in each layer.",
                default=128,
                options={
                    32: {"display_name": "32", "value": 32},
                    64: {"display_name": "64", "value": 64},
                    128: {"display_name": "128", "value": 128},
                    256: {"display_name": "256", "value": 256},
                },
            ),
            OptionsSpace[int](
                name="n_layers",
                display_name="Number of Encoder Layers",
                description="Number of encoder layers for the tabular transformer.",
                default=2,
                options={
                    1: {"display_name": "1", "value": 1},
                    2: {"display_name": "2", "value": 2},
                    3: {"display_name": "3", "value": 3},
                    4: {"display_name": "4", "value": 4},
                    5: {"display_name": "5", "value": 5},
                },
                # TODO: Check if use options or IntegerRangeSpace; AutoGluon uses categories
            ),
            IntegerRangeSpace(
                name="feature_dim",
                display_name="Fully Connected Layer Size",
                description="Size of fully connected layer in TabNet.",
                default=64,
                min=1,
                # TODO check constraints
                # AutoGluon Options: Int(8, 128)
            ),
            OptionsSpace[int](
                name="num_output_layers",
                display_name="Number of Output Layers",
                description="How many fully-connected layers on top of transformer to produce predictions.",
                default=1,
                options={
                    1: {"display_name": "1", "value": 1},
                    2: {"display_name": "2", "value": 2},
                    3: {"display_name": "3", "value": 3},
                },
                # TODO: Check if use options or IntegerRangeSpace; AutoGluon uses categories
            ),
            # The following parameters cannot be searched during HPO; see
            #   https://github.com/autogluon/autogluon/blob/8b2572e43585cbfb07e42d362f4ae8641a0fe454/tabular/src/autogluon/tabular/models/tab_transformer/hyperparameters/parameters.py#L3
            # TODO: Check if more parameters should be added
            IntegerParameter(
                name="max_emb_dim",
                display_name="Maximum Number of Embeddings",
                description="Maximum allowable amount of embeddings.",
                default=8,
                min=1,
                # TODO check constraints
            ),
            FloatParameter(
                name="aug_mask_prob",
                display_name="Augmentation Percentage",
                description="What percentage of values to apply augmentation to.",
                default=0.4,
                min=0.0,
                # TODO check constraints
            ),
            IntegerParameter(
                name="num_augs",
                display_name="Additional Augmentations",
                description="Number of augmentations to add.",
                default=0,
                min=0,
                # TODO check constraints
            ),
            IntegerParameter(
                name="num_epochs",
                display_name="Number of Epochs",
                description="The number of epochs for training the model with labeled data.",
                default=200,
                min=1,
                # TODO check constraints (maximum)
            ),
            IntegerParameter(
                name="pretrain_epochs",
                display_name="Number of Pre-Training Epochs",
                description="The number of epochs for pre-training the model with unlabeled data.",
                default=200,
                min=1,
                # TODO check constraints (maximum)
            ),
            IntegerParameter(
                name="epochs_wo_improve",
                display_name="Termination Epochs",
                description="""How many epochs to continue running without improving on metric, aka "Early Stopping Patience".""",
                default=30,
                min=1,
                # TODO check constraints (maximum)
            ),
            IntegerParameter(
                name="num_workers",
                display_name="Number of Workers",
                description="How many workers to use for torch DataLoader.",
                default=16,
                hidden=True,
                min=1,
                # TODO check constraints (maximum)
            ),
            IntegerParameter(
                name="max_columns",
                display_name="Maximum Number of Columns",
                description="Maximum number of columns TabTransformer will accept as input. This is to combat huge memory requirements/errors.",
                default=500,
                hidden=True,
                min=1,
                # TODO check constraints (maximum)
            ),
        ]

        return Parameters(
            name=Model.tabpfn,
            display_name="TabPFN",
            parameters=parameters,
        )

    @classmethod
    def get_autogluon_name(cls) -> Union[str, Type[AbstractModel]]:
        from actableai.classification.models import TabPFNModel

        return TabPFNModel
