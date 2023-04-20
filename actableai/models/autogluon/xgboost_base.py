from typing import List

from actableai.parameters.base import BaseParameter
from actableai.parameters.numeric import IntegerParameter, IntegerRangeSpace
from actableai.parameters.options import OptionsSpace


def get_parameters(problem_type: str, num_class: int) -> List[BaseParameter]:
    """
    Define parameters that are common for all variants of XGBoost boosters (tree
    and linear)

    Args:
        problem_type: Defines the type of the problem (e.g. regression,
            binary classification, etc.). See supported_problem_types variable
            for list of accepted strings
        num_class: The number of classes, used for multi-class
                classification.

    Returns:
        parameters: list containing the defined parameters
    """

    # See https://xgboost.readthedocs.io/en/latest/parameter.html,
    #   https://xgboost.readthedocs.io/en/stable/python/python_api.html,
    #   https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/xgboost/hyperparameters/parameters.py
    # For AutoGluon hyperparameter search spaces, see
    #   https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/xgboost/hyperparameters/searchspaces.py

    # TODO: Check if any additional parameters should be added

    parameters = []

    if problem_type == "regression":
        parameters = [
            OptionsSpace[str](
                name="objective",
                display_name="Objective",
                description="The learning objective used to optimize the model.",
                default=["reg:squarederror"],
                is_multi=True,
                options={
                    "reg:squarederror": {
                        "display_name": "Squared loss",
                        "value": "reg:squarederror",
                    },
                    # "reg:squaredlogerror": {
                    #     "display_name": "Squared log loss",
                    #     "value": "reg:squaredlogerror",
                    # },  # NOTE: All input labels are required to be greater than -1. Also, see metric rmsle for possible issue with this objective.
                    "reg:logistic": {
                        "display_name": "Logistic regression",
                        "value": "reg:logistic",
                    },
                    "reg:pseudohubererror": {
                        "display_name": "Pseudo Huber loss",
                        "value": "reg:pseudohubererror",
                    },
                    "reg:absoluteerror": {
                        "display_name": "L1 error",
                        "value": "reg:absoluteerror",
                    },
                    "reg:quantileerror": {
                        "display_name": "Quantile (pinball) loss",
                        "value": "reg:quantileerror",
                    },
                    "reg:gamma": {
                        "display_name": "Gamma regression with log-link",
                        "value": "reg:gamma",
                    },
                    "reg:tweedie": {
                        "display_name": "Tweedie regression with log-link",
                        "value": "reg:tweedie",
                    },
                },
                # TODO: Check if add more metrics
            ),
            # OptionsSpace[str](
            #     name="eval_metric",
            #     display_name="Evaluation Metric",
            #     description="Evaluation metric(s) for validation data.",
            #     default=["rmse"],
            #     is_multi=True,
            #     options={
            #         "rmse": {
            #             "display_name": "Root Mean Square Error (RMSE)",
            #             "value": "rmse",
            #         },
            #         # "rmsle": {
            #         #     "display_name": "Root Mean Square Log Error (RMSLE)",
            #         #     "value": "rmsle",
            #         # },  # NOTE: Default metric of reg:squaredlogerror objective. This metric reduces errors generated by outliers in dataset. But because log function is employed, rmsle might output nan when prediction value is less than -1. See reg:squaredlogerror for other requirements.
            #         "mae": {
            #             "display_name": "Mean Absolute Error (MAE)",
            #             "value": "mae",
            #         },
            #         "mape": {
            #             "display_name": "Mean Absolute Percentage Error",
            #             "value": "mape",
            #         },
            #         "mphe": {
            #             "display_name": "Mean Pseudo Huber error",
            #             "value": "mphe",
            #         },
            #     },
            #     # TODO: Check if include other metrics
            #     # TODO: Check if this should be made available
            #     # NOTE: From doc - Python users: remember to pass the metrics in as
            #     # list of parameters pairs instead of map, so that latter
            #     # eval_metric won’t override previous one
            # ),
            IntegerRangeSpace(
                name="n_estimators",
                display_name="Number of Gradient Boosted Trees",
                description="The number of gradient boosted trees, equivalent to the number of boosting rounds.",
                default=10000,
                min=1,
                # TODO check constraints
            ),
            OptionsSpace[int](
                name="proc.max_category_levels",
                display_name="Maximum Number of Levels for Categorical Features",
                description="Maximum number of allowed levels per categorical feature",
                default=[100],
                is_multi=True,
                options={
                    10: {"display_name": "10", "value": 10},
                    100: {"display_name": "100", "value": 100},
                    200: {"display_name": "200", "value": 200},
                    300: {"display_name": "300", "value": 300},
                    400: {"display_name": "400", "value": 400},
                    500: {"display_name": "500", "value": 500},
                    1000: {"display_name": "1000", "value": 1000},
                    10000: {"display_name": "10000", "value": 10000},
                },
                # TODO: Check if should use options or integer space
            ),
        ]

    elif problem_type == "binary":
        parameters = [
            OptionsSpace[str](
                name="objective",
                display_name="Objective",
                description="The learning objective used to optimize the model.",
                default=["binary:logistic"],
                is_multi=True,
                options={
                    "binary:logistic": {
                        "display_name": "Logistic Regression (probability)",
                        "value": "binary:logistic",
                    },
                    "binary:logitraw": {
                        "display_name": "Logistic Regression (score before logistic transformation)",
                        "value": "binary:logitraw",
                    },
                    "binary:hinge": {
                        "display_name": "Hinge loss (output 0/1)",
                        "value": "binary:hinge",
                    },
                },
                # TODO: Check if add more metrics
            ),
            # OptionsSpace[str](
            #     name="eval_metric",
            #     display_name="Evaluation Metric",
            #     description="Evaluation metric(s) for validation data.",
            #     default=["logloss"],
            #     is_multi=True,
            #     options={
            #         "logloss": {
            #             "display_name": "Negative log-likelihood (log loss)",
            #             "value": "logloss",
            #         },
            #         "error": {
            #             "display_name": "Error rate",
            #             "value": "error",
            #         },
            #         "auc": {
            #             "display_name": "Receiver Operating Characteristic Area under the Curve",
            #             "value": "auc",
            #         },  # NOTE: the objective should be binary:logistic or similar functions that work on probability.
            #         "aucpr": {
            #             "display_name": "Area under the PR Curve",
            #             "value": "aucpr",
            #         },
            #         "ndcg": {
            #             "display_name": "Normalized Discounted Cumulative Gain (NDCG)",
            #             "value": "ndcg",
            #         },
            #         "map": {
            #             "display_name": "Mean Average Precision (MAP)",
            #             "value": "map",
            #         },
            #     },
            #     # TODO: Check if include other metrics
            #     # TODO: Check if this should be made available
            #     # NOTE: From doc - Python users: remember to pass the metrics in as
            #     # list of parameters pairs instead of map, so that latter
            #     # eval_metric won’t override previous one
            # ),
        ]

    elif problem_type == "multiclass":
        parameters = [
            OptionsSpace[str](
                name="objective",
                display_name="Objective",
                description="The learning objective used to optimize the model.",
                default=["multi:sofprob"],
                is_multi=True,
                options={
                    "multi:softmax": {
                        "display_name": "Softmax",
                        "value": "multi:softmax",
                    },
                    "multi:softprob": {
                        "display_name": "Softmax probabilities per data point per class",
                        "value": "multi:softprob",
                    },
                    "rank:map": {
                        "display_name": "Maximize Mean Average Precision (MAP)",
                        "value": "rank:map",
                    },
                    "rank:pairwise": {
                        "display_name": "Minimize pairwise loss",
                        "value": "rank:pairwise",
                    },
                    "rank:ndcg": {
                        "display_name": "Maximize Normalized Discounted Cumulative Gain (NDCG)",
                        "value": "rank:ndcg",
                    },
                },
                # Check if add more metrics, and if can use rank-based metrics
            ),
            IntegerParameter(
                name="num_class",
                display_name="Number of Classes",
                description="Number of classes of the target variable.",
                default=num_class,
                min=1,
                hidden=True,
                # TODO: Check if should be hidden or visible
            ),
            # OptionsSpace[str](
            #     name="eval_metric",
            #     display_name="Evaluation Metric",
            #     description="Evaluation metric(s) for validation data.",
            #     default=["mlogloss"],
            #     is_multi=True,
            #     options={
            #         "mlogloss": {
            #             "display_name": "Negative log-likelihood (log loss)",
            #             "value": "mlogloss",
            #         },
            #         "merror": {
            #             "display_name": "Error rate",
            #             "value": "merror",
            #         },
            #         "auc": {
            #             "display_name": "Receiver Operating Characteristic Area under the Curve",
            #             "value": "auc",
            #         },  # NOTE: When used with multi-class classification, objective should be multi:softprob instead of multi:softmax, as the latter doesn’t output probability. Also the AUC is calculated by 1-vs-rest with reference class weighted by class prevalence.
            #         "aucpr": {
            #             "display_name": "Area under the PR Curve",
            #             "value": "aucpr",
            #         },
            #         "ndcg": {
            #             "display_name": "Normalized Discounted Cumulative Gain",
            #             "value": "ndcg",
            #         },
            #         "map": {
            #             "display_name": "Mean Average Precision",
            #             "value": "map",
            #         },
            #     },
            #     # TODO: Check if include other metrics
            #     # TODO: Check if this should be made available
            #     # NOTE: From doc - Python users: remember to pass the metrics in as
            #     # list of parameters pairs instead of map, so that latter
            #     # eval_metric won’t override previous one
            # ),
        ]

    return parameters
