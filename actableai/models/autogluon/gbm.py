from actableai.models.autogluon.base import BaseParams, Model
from actableai.parameters.boolean import BooleanSpace
from actableai.parameters.numeric import (
    IntegerRangeSpace,
    FloatRangeSpace,
    IntegerParameter,
)
from actableai.parameters.options import OptionsSpace
from actableai.parameters.parameters import Parameters


class GBMParams(BaseParams):
    """Parameter class for GBM Model."""

    supported_problem_types = ["regression", "binary", "multiclass"]
    _autogluon_name = "GBM"
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

        # See https://lightgbm.readthedocs.io/en/latest/Parameters.html,
        #   https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/lgb/hyperparameters/parameters.py
        # For AutoGluon search spaces, see
        # https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/lgb/hyperparameters/searchspaces.py
        # TODO: Check if any additional parameters should be added
        parameters = [
            FloatRangeSpace(
                name="learning_rate",
                display_name="Learning Rate",
                description="Learning rate used during training.",
                default=(0.03, 0.05),
                min=0.0001,
                _log=True,
                # TODO check constraints
                # NOTE: AutoGluon sets log=True (log-scale)
            ),
            BooleanSpace(
                name="extra_trees",
                display_name="Extra Trees",
                description="Whether to use extremely randomized trees.",
                default="false",
            ),
            IntegerRangeSpace(
                name="max_depth",
                display_name="Maximum Depth",
                description="Limit for the maximum depth of the model. A value of -1 indicates no limit.",
                default=-1,
                min=-1,
                # TODO check constraints
            ),
            IntegerRangeSpace(
                name="min_data_in_leaf",
                display_name="Minimum Data in Leaf",
                description="Minimal number of data in one leaf. Can be used to deal with over-fitting. Note that the actual value of a split may be less than the specified value.",
                default=(3, 20),
                min=0,
                # TODO check constraints (maximum)
            ),
            OptionsSpace[str](
                name="boosting",
                display_name="Gradient Boosting Method",
                description="Gradient boosting method to use. Note: GBDT is used for the first 1/learning_rate iterations.",
                default=["gbdt"],
                is_multi=True,
                options={
                    "gbdt": {
                        "display_name": "Gradient Boosted Decision Tree (GBDT)",
                        "value": "gbdt",
                    },
                    "rf": {"display_name": "Random Forest", "value": "rf"},
                    "dart": {
                        "display_name": "Dropouts meet Multiple Additive Regression Trees (DART)",
                        "value": "dart",
                    },
                },
            ),
            FloatRangeSpace(
                name="lambda_l2",
                display_name="L2 Regularization",
                description="L2 regularization used during training.",
                default=0.0,
                min=0,
                # TODO check constraints (maximum)
            ),
            FloatRangeSpace(
                name="lambda_l1",
                display_name="L1 Regularization",
                description="L1 regularization used during training.",
                default=0.0,
                min=0,
                # TODO check constraints (maximum)
            ),
            FloatRangeSpace(
                name="linear_lambda",
                display_name="Linear Tree Regularization",
                description="Linear tree regularization used during training.",
                default=0.0,
                min=0,
                # TODO check constraints (maximum)
            ),
            FloatRangeSpace(
                name="feature_fraction",
                display_name="Fraction of Feature Subset",
                description="Randomly select a subset of features on each iteration (tree) if value is smaller than 1. For example, if value is 0.8, 80% of features will be selected before training each tree. Can be used to deal with over-fitting and to speed up training of this model.",
                default=(0.9, 1.0),
                min=0.0001,  # NOTE: Minimum should be >0
                max=1.00001,  # NOTE: Max should be <=1.0
            ),
            IntegerRangeSpace(
                name="num_leaves",
                display_name="Maximum Number of Leaves in a Tree",
                description="The maximum number of leaves in a tree.",
                default=(31, 128),
                min=2,  # NOTE: Minimum should be >1
                max=131072.0001,  # NOTE: Max should be <=131072
            ),
        ]

        # Parameters specific for each problem type:
        if problem_type == "regression":
            parameters += [
                OptionsSpace[str](
                    name="objective",
                    display_name="Objective Metric",
                    description="The objective metric to be optimized.",
                    default=["regression"],
                    is_multi=True,
                    options={
                        "regression": {
                            "display_name": "L2 (MSE)",
                            "value": "regression",
                        },
                        "regression_l1": {
                            "display_name": "L1 (MAE)",
                            "value": "regression_l1",
                        },
                        "quantile": {
                            "display_name": "Quantile",
                            "value": "quantile",
                        },
                        "mape": {"display_name": "MAPE", "value": "mape"},
                        "huber": {"display_name": "Huber", "value": "huber"},
                        "fair": {"display_name": "fair", "value": "fair"},
                    },
                    # TODO: Check if add any additional metrics
                ),
                OptionsSpace[str](
                    name="metric",
                    display_name="Evaluation Metrics",
                    description="Metric(s) to be evaluated on the evaluation set.",
                    default=[""],
                    is_multi=True,
                    options={
                        "": {
                            "display_name": "Metric corresponding to specified 'Objective Metric'",
                            "value": "",
                        },
                        "None": {"display_name": "No Metric", "value": "None"},
                        "l1": {"display_name": "L1 (MAE)", "value": "l1"},
                        "l2": {"display_name": "L2 (MSE)", "value": "l2"},
                        "rmse": {"display_name": "RMSE", "value": "rmse"},
                        "quantile": {
                            "display_name": "Quantile",
                            "value": "quantile",
                        },
                        "mape": {"display_name": "MAPE", "value": "mape"},
                        "huber": {"display_name": "Huber", "value": "huber"},
                        "fair": {"display_name": "fair", "value": "fair"},
                    },
                    # TODO check available options
                    # TODO: Check any additional metrics
                ),
            ]

        elif problem_type == "binary":
            parameters += [
                OptionsSpace[str](
                    name="objective",
                    display_name="Objective Metric",
                    description="The objective metric to be optimized.",
                    default=["binary"],
                    is_multi=True,
                    options={
                        "binary": {
                            "display_name": "Binary log loss",
                            "value": "binary",
                        },  # NOTE: requires labels in {0, 1}
                        "cross_entropy": {
                            "display_name": "Cross-entropy",
                            "value": "cross_entropy",
                        },  # NOTE: requires labels in {0, 1}
                    },
                    # TODO: Check if add any additional metrics
                ),
                OptionsSpace[str](
                    name="metric",
                    display_name="Evaluation Metrics",
                    description="Metric(s) to be evaluated on the evaluation set.",
                    default=[""],
                    is_multi=True,
                    options={
                        "": {
                            "display_name": "Metric corresponding to specified 'Objective Metric'",
                            "value": "",
                        },
                        "None": {"display_name": "No Metric", "value": "None"},
                        "map": {"display_name": "MAP", "value": "map"},
                        "auc": {"display_name": "AUC", "value": "auc"},
                        "average_precision": {
                            "display_name": "Average precision score",
                            "value": "average_precision",
                        },
                        "binary_logloss": {
                            "display_name": "Log loss",
                            "value": "binary_logloss",
                        },
                        "binary_error": {
                            "display_name": "Error rate (0 for correct, 1 for incorrect)",
                            "value": "binary_error",
                        },
                        "auc_mu": {"display_name": "AUC-mu", "value": "auc_mu"},
                    },
                    # TODO check if available options valid for problem type
                    # TODO: Check any additional metrics
                ),
            ]

        elif problem_type == "multiclass":
            parameters += [
                OptionsSpace[str](
                    name="objective",
                    display_name="Objective Metric",
                    description="The objective metric to be optimized.",
                    default=["multiclass"],
                    is_multi=True,
                    options={
                        "multiclass": {
                            "display_name": "Softmax objective function",
                            "value": "multiclass",
                        },
                        "multiclassanova": {
                            "display_name": "One-vs-all binary objective function",
                            "value": "multiclassanova",
                        },
                    },
                    # TODO: Check if add any additional metrics
                ),
                IntegerParameter(
                    name="num_class",
                    display_name="Number of Classes",
                    description="Number of classes of the target variable.",
                    default=num_class,
                    min=1,  # NOTE: Minimum should be >0
                    hidden=True,
                    # TODO: Check if should be hidden or visible
                ),
                OptionsSpace[str](
                    name="metric",
                    display_name="Evaluation Metrics",
                    description="Metric(s) to be evaluated on the evaluation set.",
                    default=[""],
                    is_multi=True,
                    options={
                        "": {
                            "display_name": "Metric corresponding to specified 'Objective Metric'",
                            "value": "",
                        },
                        "None": {"display_name": "No Metric", "value": "None"},
                        "map": {"display_name": "MAP", "value": "map"},
                        "auc": {"display_name": "AUC", "value": "auc"},
                        "average_precision": {
                            "display_name": "Average precision score",
                            "value": "average_precision",
                        },
                        "auc_mu": {"display_name": "AUC-mu", "value": "auc_mu"},
                        "multi_logloss": {
                            "display_name": "Log loss",
                            "value": "multi_logloss",
                        },
                        "multi_error": {
                            "display_name": "Error rate",
                            "value": "multi_error",
                        },
                        "cross_entropy": {
                            "display_name": "Cross-entropy",
                            "value": "cross_entropy",
                        },
                    },
                    # TODO check if available options are valid for problem type
                    # TODO: Check any additional metrics
                ),
            ]

        return Parameters(
            name=Model.gbm,
            display_name="GBM",
            parameters=parameters,
        )
