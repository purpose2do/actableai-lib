from actableai.models.autogluon.base import BaseParams, Model
from actableai.parameters.numeric import IntegerRangeSpace, FloatRangeSpace
from actableai.parameters.options import OptionsSpace
from actableai.parameters.parameters import Parameters
from .xgboost_base import get_parameters


class XGBoostTreeParams(BaseParams):
    """Parameter class for XGBoost Model when using a tree booster."""

    supported_problem_types = ["regression", "binary", "multiclass"]
    _autogluon_name = "XGB"
    explain_samples_supported = True

    @classmethod
    def _get_hyperparameters(
        cls, *, problem_type: str, num_class: int, **kwargs
    ) -> Parameters:
        """Returns the hyperparameters space of the model.
        Args:
            problem_type: Defines the type of the problem (e.g. regression,
                binary classification, etc.). See
                cls.supported_problem_types
                for list of accepted strings
            num_class: The number of classes, used for multi-class
                classification.

        Returns:
            The hyperparameters space.
        """
        # Get parameters common to all variants:
        parameters = get_parameters(problem_type=problem_type, num_class=num_class)

        # See https://xgboost.readthedocs.io/en/latest/parameter.html,
        #   https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/xgboost/hyperparameters/parameters.py
        # For AutoGluon hyperparameter search spaces, see
        #   https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/xgboost/hyperparameters/searchspaces.py
        # TODO: Check if any additional parameters should be added
        parameters += [
            # Tree Booster-specific parameters:
            OptionsSpace[str](
                name="booster",
                display_name="Booster",
                description="Which booster to use.",
                default=["gbtree"],
                is_multi=True,
                options={
                    "gbtree": {"display_name": "GBTree", "value": "gbtree"},
                    "dart": {"display_name": "Dart", "value": "dart"},
                },
            ),
            FloatRangeSpace(
                name="learning_rate",
                display_name="Learning Rate",
                description="Learning rate used during training.",
                default=0.1,
                min=0.0001,
                max=1.0001,
                _log=True,
                # TODO check constraints
                # NOTE: AutoGluon sets log=True (log-scale)
            ),
            FloatRangeSpace(
                name="gamma",
                display_name="Gamma",
                description="Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger the value, the more conservative the algorithm will be.",
                default=0,
                min=0,
                # NOTE: HPO disabled in AutoGluon since it made search worse
            ),
            IntegerRangeSpace(
                name="max_depth",
                display_name="Maximum Depth",
                description="Maximum depth of a tree. Increasing this value will make the model more complex and more likely to over-fit. A value of 0 indicates no limit on depth.",
                default=6,
                min=0,
                max=10,
                # TODO: check constraints; it is stated in the docs that "XGBoost
                # aggressively consumes memory when training a deep tree"
                # NOTE: 'exact' tree method requires non-zero value
            ),
            FloatRangeSpace(
                name="min_child_weight",
                display_name="Minimum sum of instance weight in a child",
                description="Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger the value, the more conservative the algorithm will be.",
                default=1,
                min=0,
            ),
            FloatRangeSpace(
                name="max_delta_step",
                display_name="Maximum delta step",
                description="Maximum delta step we allow each leaf output to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update.",
                default=0,
                min=0,
            ),
            FloatRangeSpace(
                name="subsample",
                display_name="Subsample Ratio",
                description="Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent over-fitting. Sub-sampling will occur once in every boosting iteration.",
                default=1,
                min=0.0001,  # Minimum should be >0
                max=1.0001,  # Maximum should be <=1
                # NOTE: HPO disabled in AutoGluon since it made search worse
            ),
            FloatRangeSpace(
                name="alpha",
                display_name="L1 Regularization Term",
                description="L1 regularization term on weights. Increasing this value will make model more conservative.",
                default=0,
                min=0,
                # NOTE: HPO disabled in AutoGluon since it made search worse
            ),
            FloatRangeSpace(
                name="lambda",
                display_name="L2 Regularization Term",
                description="L2 regularization term on weights. Increasing this value will make model more conservative.",
                default=1,
                min=0,
                # NOTE: HPO disabled in AutoGluon since it made search worse
            ),
            FloatRangeSpace(
                name="colsample_bytree",
                display_name="Column Sub-sampling for Each Tree",
                description="The subsample ratio of columns when constructing each tree. Sub-sampling occurs once for every tree constructed.",
                default=1,
                min=0.0001,  # Minimum should be >0
                max=1.0001,  # Maximum should be <=1
            ),
        ]

        return Parameters(
            name=Model.xgb_tree,
            display_name="XGBoost (Tree Booster)",
            parameters=parameters,
        )
