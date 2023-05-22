from actableai.models.autogluon.base import BaseParams, Model
from actableai.parameters.numeric import IntegerRangeSpace, FloatRangeSpace
from actableai.parameters.parameters import Parameters


class CATParams(BaseParams):
    """Parameter class for CAT Model."""

    supported_problem_types = [
        "regression",
        "binary",
        "multiclass",
    ]  # NOTE: "softclass" can also be supported
    _autogluon_name = "CAT"
    explain_samples_supported = True

    @classmethod
    def _get_hyperparameters(
        cls, *, problem_type: str, device: str, **kwargs
    ) -> Parameters:
        """Returns the hyperparameters space of the model.

        Args:
            problem_type: Defines the type of the problem (e.g. regression,
                binary classification, etc.). See
                cls.supported_problem_types
                for list of accepted strings [unused]
            device: Which device is being used, can be one of 'cpu' or 'gpu'

        Returns:
            The hyperparameters space.
        """
        # See https://catboost.ai/en/docs/concepts/parameter-tuning,
        #   https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/catboost/hyperparameters/parameters.py
        # For AutoGluon search spaces, see
        # https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/catboost/hyperparameters/searchspaces.py
        # TODO: Check if any additional parameters should be added
        # NOTE: CatBoost package also enables hyperparameter search

        parameters = [
            FloatRangeSpace(
                name="learning_rate",
                display_name="Learning Rate",
                description="Learning rate used during training.",
                default=0.05,
                min=0.0001,
                _log=True,
                # TODO check constraints
                # NOTE: AutoGluon sets log=True (log-scale)
            ),
            IntegerRangeSpace(
                name="iterations",
                display_name="Maximum Number of Trees",
                description="The maximum number of trees that can be built, limiting the number of iterations.",
                default=10000,
                min=1,
                # TODO check constraints
                # TODO: check if should be included; AutoGluon seems to only use
                # value of 10,000
            ),
            # IntegerRangeSpace(
            #     name="num_leaves",
            #     display_name="Maximum Number of Leaves",
            #     description="The maximum number of leaves in the resulting tree.",
            #     default=(1, 250),
            #     min=1,
            #     # TODO check constraints
            #     # NOTE: Can be used only with the Lossguide growing policy.
            # ),
            # IntegerRangeSpace(
            #     name="min_child_samples",
            #     display_name="Minimum Number of Training Samples in a Leaf",
            #     description="The minimum number of training samples in a leaf. CatBoost does not search for new splits in leaves with samples count less than the specified value.",
            #     default=(1, 64),
            #     min=1,
            #     max=65.0001,  # 64.0001
            #     # TODO check constraints
            #     # NOTE: Can be used only with the Lossguide and Depthwise growing policies.
            # ),
            # FloatRangeSpace(
            #     name="l2_leaf_reg",
            #     display_name="L2 Regularization",
            #     description="Coefficient at the L2 regularization term of the cost function.",
            #     default=3,
            #     min=0,
            #     # TODO check constraints (maximum)
            # ),
            # OptionsSpace[str](
            #     name="grow_policy",
            #     display_name="Tree Growing Policy",
            #     description="The tree growing policy. Defines how to perform greedy tree construction.",
            #     default=["SymmetricTree"],
            #     is_multi=True,
            #     options={
            #         "SymmetricTree": {
            #             "display_name": "Symmetric Tree",
            #             "value": "SymmetricTree",
            #         },
            #         "Depthwise": {
            #             "display_name": "Depth-wise",
            #             "value": "Depthwise",
            #         },
            #         "Lossguide": {
            #             "display_name": "Loss-guided",
            #             "value": "Lossguide",
            #         },
            #     },
            #     # NOTE: For Depthwise and Lossguide: Models with this growing
            #     # policy can not be analyzed using the PredictionDiff feature
            #     # importance and can be exported only to json and cbm.
            # ),
            IntegerRangeSpace(
                name="depth",
                display_name="Tree Depth",
                description="Depth of a tree.",
                default=(1, 8),
                min=1,
                max=17 if device == "cpu" else 9,  # Excluded
                # NOTE:
                # CPU — Any integer up to  16
                # GPU — Any integer up to 8 pairwise modes (YetiRank,
                #   PairLogitPairwise and QueryCrossEntropy) and up to 16 for
                #   all other loss functions.
                # TODO: Check if should adjust max depending on mode
            ),
        ]

        return Parameters(
            name=Model.cat,
            display_name="CATBoost",
            parameters=parameters,
        )
