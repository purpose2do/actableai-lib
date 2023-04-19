from actableai.models.autogluon.base import BaseParams, Model
from actableai.parameters.numeric import FloatRangeSpace
from actableai.parameters.options import OptionsSpace, OptionsParameter
from actableai.parameters.parameters import Parameters
from .xgboost_base import get_parameters


class XGBoostLinearParams(BaseParams):
    """Parameter class for XGBoost Model when using a linear booster."""

    # TODO: This model is currently disabled since it does not run; re-enable
    # when this is fixed
    supported_problem_types = []  # ["regression", "binary", "multiclass"]
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
            # Linear Booster-specific parameters:
            OptionsParameter[str](
                name="booster",
                display_name="Booster",
                description="Which booster to use.",
                default="gblinear",
                is_multi=False,
                hidden=True,
                options={
                    "gblinear": {"display_name": "GBLinear", "value": "gblinear"},
                },
            ),
            FloatRangeSpace(
                name="lambda",
                display_name="L2 regularization term",
                description="L2 regularization term on weights. Increasing this value will make model more conservative. Normalized to number of training examples.",
                default=0,
                min=0,
                # TODO check default range, constraints
                # NOTE: HPO disabled in AutoGluon since it made search worse
            ),
            FloatRangeSpace(
                name="alpha",
                display_name="L1 regularization term",
                description="L1 regularization term on weights. Increasing this value will make model more conservative. Normalized to number of training examples.",
                default=0,
                min=0,
                # TODO check default range, constraints
                # NOTE: HPO disabled in AutoGluon since it made search worse
            ),
            OptionsSpace[str](
                name="updater",
                display_name="Updater",
                description="Choice of algorithm to fit linear model.",
                default=["shotgun"],
                is_multi=True,
                options={
                    "shotgun": {
                        "display_name": "Shotgun",
                        "value": "shotgun",
                    },
                    "coord_descent": {
                        "display_name": "Coordinate Descent",
                        "value": "coord_descent",
                    },
                },
            ),
        ]

        return Parameters(
            name=Model.xgb_linear,
            display_name="XGBoost (Linear Booster)",
            parameters=parameters,
        )
