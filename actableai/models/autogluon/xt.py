from actableai.models.autogluon.base import BaseParams, Model
from actableai.parameters.numeric import FloatParameter, IntegerParameter
from actableai.parameters.options import OptionsParameter
from actableai.parameters.parameters import Parameters


class XTParams(BaseParams):
    """Parameter class for XT Model."""

    # NOTE: "quantile" also supported, but temporarily disabled due to
    # potentially high memory consumption
    supported_problem_types = ["regression", "binary", "multiclass"]
    _autogluon_name = "XT"
    explain_samples_supported = True

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
        #   Classification: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
        #   Regression: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html
        # NOTE: Parameters virtually identical to those for Random Forest (RF).
        # TODO: Consider moving to common file.
        # TODO: Check if any additional parameters should be added
        # TODO: Check if used only for classification or if it can also be used
        # for regression
        # NOTE: Hyperparameter tuning is disabled for this model in AutoGluon.
        parameters = [
            IntegerParameter(
                name="n_estimators",
                display_name="Number of Trees",
                description="Number of trees in the forest.",
                default=100,
                min=1,
                # TODO check constraints (include max?)
            ),
            # IntegerParameter(
            #     name="max_depth",
            #     display_name="Maximum Tree Depth",
            #     description="Maximum depth of a tree.",
            #     default=100,
            #     min=1,
            #     max=10001,
            #     # TODO check constraints (include max?)
            #     # TODO: check if should include; by default, uses a value of
            #     # None to expand all nodes until all leaves are pure or until
            #     # all leaves contain less than min_samples_split samples
            # ),
            IntegerParameter(
                name="min_samples_split",
                display_name="Minimum Samples to Split Node",
                description="The minimum number of samples required to split an internal node.",
                default=2,
                min=1,
                max=10001,
                # TODO check constraints (include max?)
                # NOTE/TODO: This parameter also supports the use of a float,
                # representing the fraction of samples
            ),
            IntegerParameter(
                name="min_samples_leaf",
                display_name="Minimum Samples at a Leaf Node",
                description="The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches, where min_samples_leaf is the value of this parameter. This may have the effect of smoothing the model, especially in regression.",
                default=1,
                min=1,
                max=1001,
                # TODO check constraints (include max?)
                # NOTE/TODO: This parameter also supports the use of a float,
                # representing the fraction of samples
            ),
            OptionsParameter[str](
                name="max_features",
                display_name="Number of Features for Best Split",
                description="The number of features to consider when looking for the best split. 'sqrt' = sqrt(number of features), 'log2' = log2(number of features). Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than the set number of features",
                default="sqrt",
                is_multi=False,
                options={
                    "sqrt": {
                        "display_name": "sqrt",
                        "value": "sqrt",
                    },
                    "log2": {
                        "display_name": "log2",
                        "value": "log2",
                    },
                },
                # NOTE: This parameter also supports int and float inputs
                # TODO: Check if 'None' can be used as a string. Also ensure
                # that description is also updated.
            ),
            FloatParameter(
                name="min_impurity_decrease",
                display_name="Minimum Impurity Decrease",
                description="A node will be split if this split induces a decrease of the impurity greater than or equal to this value.",
                default=0,
                min=0,
                # TODO check constraints (include max?)
            ),
        ]

        if problem_type in ["regression", "quantile"]:
            parameters += [
                OptionsParameter[str](
                    name="criterion",
                    display_name="Quality of Split Criterion",
                    description="The function to measure the quality of a split.",
                    default="squared_error",
                    is_multi=False,
                    options={
                        "squared_error": {
                            "display_name": "Mean Squared Error (MSE)",
                            "value": "squared_error",
                        },
                        "absolute_error": {
                            "display_name": "Mean Absolute Error (MAE)",
                            "value": "absolute_error",
                        },
                        "friedman_mse": {
                            "display_name": "MSE with Friedman's improvement score",
                            "value": "friedman_mse",
                        },
                        "poisson": {
                            "display_name": "Poisson deviance",
                            "value": "poisson",
                        },
                    },
                ),
            ]
        else:  # Binary/multi-class Classification
            parameters += [
                OptionsParameter[str](
                    name="criterion",
                    display_name="Quality of Split Criterion",
                    description="The function to measure the quality of a split.",
                    default=["gini", "entropy"],
                    is_multi=False,
                    options={
                        "gini": {"display_name": "Gini impurity", "value": "gini"},
                        "entropy": {
                            "display_name": "Entropy",
                            "value": "entropy",
                        },
                        "log_loss": {
                            "display_name": "Log-loss",
                            "value": "log_loss",
                        },
                    },
                ),
            ]

        return Parameters(
            name=Model.xt,
            display_name="Extra Trees",
            parameters=parameters,
        )
