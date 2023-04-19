from actableai.models.autogluon.base import BaseParams, Model
from actableai.parameters.boolean import BooleanParameter
from actableai.parameters.numeric import IntegerParameter, FloatParameter
from actableai.parameters.options import OptionsParameter
from actableai.parameters.parameters import Parameters


class LRParams(BaseParams):
    """Parameter class for LR Model."""

    # TODO: Check supported problem types. Also check if quantile regression is supported
    supported_problem_types = ["regression", "binary", "multiclass"]
    _autogluon_name = "LR"
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
        # https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/lr/hyperparameters/parameters.py,
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        # TODO: Check if any additional parameters should be added, such as the
        # text-based features
        # TODO: Check if runs ok for classification (binary + multi-class);
        # should work based on
        # https://github.com/autogluon/autogluon/blob/023c7e7ab3da84a50153d2d6c6a28946e939743d/tabular/src/autogluon/tabular/models/lr/lr_model.py
        # However, need to test since does not seem to be very stable (might
        # crash even with valid parameters since HPO is not supported)
        # NOTE: Hyperparameter tuning is disabled for this model in AutoGluon.
        # NOTE: ‘penalty’ parameter can be used for regression to specify regularization method: ‘L1’ and ‘L2’ values are supported.

        # NOTE: Some text/NLP-based parameters used in parameter baseline (see
        # https://github.com/autogluon/autogluon/blob/663b2c9cbaf73de3bc4cbf3db2e2930c8b37adf0/tabular/src/autogluon/tabular/models/lr/hyperparameters/parameters.py#L21)
        # TODO: Check if should include or exclude these text/NLP features, and
        # if should be included only if a parameter is set.
        # TODO: Check if/how should include proc.ngram_range (tuple),
        # proc.impute_strategy, C (not defined)
        parameters = [
            OptionsParameter[str](
                name="handle_text",
                display_name="Text Handling",
                description="How text should be handled.",
                default="ignore",
                is_multi=False,
                options={
                    "ignore": {
                        "display_name": "Ignore NLP features",
                        "value": "ignore",
                    },
                    "include": {
                        "display_name": "Use both regular and NLP features",
                        "value": "include",
                    },
                    "only": {
                        "display_name": "Only use NLP features",
                        "value": "only",
                    },
                },
            ),
            IntegerParameter(
                name="vectorizer_dict_size",
                display_name="Vectorizer Dictionary Size",
                description="Size of TF-IDF vectorizer dictionary; used only in text model.",
                default=75000,
                min=1,
                # TODO: Check constraints
            ),
            FloatParameter(
                name="proc.skew_threshold",
                display_name="Skew Threshold",
                description="Numerical features whose absolute skewness is greater than this receive special power-transform preprocessing. Choose big value to avoid using power-transforms.",
                default=0.99,
                min=0,
                # TODO: Check constraints
            ),
        ]

        # TODO: Should 'quantile' be added (not directly supported)
        if problem_type in ["regression"]:
            # See
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
            parameters += [
                # FloatParameter(
                #     name="alpha",
                #     display_name="Regularization Strength Multiplier",
                #     description="Constant that multiplies the L2 term, controlling regularization strength. When equal to 0, the objective is equivalent to ordinary least squares.",
                #     default=1.0,
                #     min=0,
                # ),
                BooleanParameter(
                    name="fit_intercept",
                    display_name="Fit Intercept?",
                    description="Whether to fit the intercept for this model. If set to false, no intercept will be used in calculations (i.e. features are expected to be centered).",
                    default="true",
                ),
                FloatParameter(
                    name="tol",
                    display_name="Solution Precision",
                    description="Precision of the solution. Note that this parameter has no effect for solvers 'SVD' and 'Cholesky'",
                    default=1e-4,
                    min=1e-6,
                    # TODO: Check constraints
                ),
                OptionsParameter[str](
                    name="solver",
                    display_name="Solver",
                    description="Solver to use in the computational routines.",
                    default="auto",
                    is_multi=False,
                    options={
                        "auto": {"display_name": "Auto", "value": "auto"},
                        # "lbfgs": {"display_name": "L-BFGS-B", "value": "lbfgs"},
                        "svd": {
                            "display_name": "Singular Value Decomposition (SVD)",
                            "value": "svd",
                        },
                        "cholesky": {
                            "display_name": "Cholesky",
                            "value": "cholesky",
                        },
                        "sparse-cg": {
                            "display_name": "Conjugate Gradient Solver",
                            "value": "sparse-cg",
                        },
                        "lsqr": {
                            "display_name": "Regularized Least-Squares Routine",
                            "value": "lsqr",
                        },
                        "sag": {
                            "display_name": "Stochastic Average Gradient Descent (SAG)",
                            "value": "sag",
                        },
                        "saga": {"display_name": "SAGA", "value": "saga"},
                    },
                    # NOTE: Note ‘sag’ and ‘saga’ fast convergence is only
                    # guaranteed on features with approximately the same scale. You
                    # can preprocess the data with a scaler from
                    # sklearn.preprocessing.
                    # NOTE: lbfgs can be used only when positive is True.
                ),
                # BooleanParameter(
                #     name="positive",
                #     display_name="Positive?",
                #     description="When set to True, forces the coefficients to be positive. Only 'L-BFGS-B' solver is supported in this case.",
                #     default="false",
                #     # NOTE: Only supported by lbfgs
                #     # TODO: Should this parameter be included?
                # ),
                OptionsParameter[str](
                    name="penalty",
                    display_name="Penalty Normalization",
                    description="Specify the normalization of the penalty.",
                    default="L2",
                    is_multi=False,
                    options={
                        "L1": {"display_name": "L1", "value": "L1"},
                        "L2": {"display_name": "L2", "value": "L2"},
                    },
                ),
            ]
        else:  # Binary/multi-class
            parameters += [
                OptionsParameter[str](
                    name="penalty",
                    display_name="Penalty Normalization",
                    description="Specify the normalization of the penalty. L1 is compatible with 'liblinear' and 'saga' solvers, elasticnet is only available with saga.",
                    default="l2",
                    is_multi=False,
                    options={
                        # "none": {"display_name": "No Penalty", "value": "none"},
                        "l1": {"display_name": "L1", "value": "l1"},
                        "l2": {"display_name": "L2", "value": "l2"},
                        "elasticnet": {
                            "display_name": "Both L1 and L2",
                            "value": "elasticnet",
                        },
                    },
                    # NOTE/TODO: Some penalties may not work with some solvers! Also
                    # check 'solver'
                ),
                OptionsParameter[str](
                    name="solver",
                    display_name="Solver",
                    description="Algorithm to use in the optimization problem. For small datasets, 'liblinear' is a good choice, whereas 'sag' and 'saga' are faster for large ones. 'newton-cholesky' is a good choice when the number of samples is much greater than the number of features, especially with one-hot encoded categorical features with rare categories. See description for 'Penalty Normalization' for compatible penalties with each solver.",
                    default="lbfgs",
                    is_multi=False,
                    options={
                        "lbfgs": {"display_name": "L-BFGS-B", "value": "lbfgs"},
                        "liblinear": {
                            "display_name": "liblinear",
                            "value": "liblinear",
                        },
                        "newton-cg": {
                            "display_name": "newton-cg",
                            "value": "newton-cg",
                        },
                        # "newton-cholesky": {
                        #    "display_name": "Newton-Cholesky",
                        #    "value": "newton-cholesky",
                        # },
                        "sag": {
                            "display_name": "Stochastic Average Gradient Descent (SAG)",
                            "value": "sag",
                        },
                        "saga": {"display_name": "SAGA", "value": "saga"},
                    },
                    # NOTE/TODO: Some penalties may not work with some penalty terms! Also
                    # check 'penalty' (see
                    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#:~:text=Warning%20The%20choice%20of%20the%20algorithm%20depends%20on%20the%20penalty%20chosen. )
                    # NOTE: 'liblinear' is limited to one-versus-rest schemes.
                    # NOTE: 'newton-cholesky' is limited to binary classification
                    # and the one-versus-rest reduction for multiclass
                    # classification. Be aware that the memory usage of this solver
                    # has a quadratic dependency on n_features because it explicitly
                    # computes the Hessian matrix.
                    # NOTE: Note ‘sag’ and ‘saga’ fast convergence is only
                    # guaranteed on features with approximately the same scale. You
                    # can preprocess the data with a scaler from
                    # sklearn.preprocessing.
                    # TODO: 'newton-cholesky' seems unstable and has been
                    # disabled for now
                ),
                FloatParameter(
                    name="tol",
                    display_name="Stopping Tolerance",
                    description="Tolerance for stopping criteria.",
                    default=1.0,
                    min=1e-6,
                    # TODO: Check constraints
                ),
                FloatParameter(
                    name="C",
                    display_name="Regularization Strength Inverse",
                    description="Inverse of regularization strength. Smaller values specify stronger regularization.",
                    default=1,
                    min=1e-6,
                    # TODO: Check constraints
                ),
                BooleanParameter(
                    name="fit_intercept",
                    display_name="Fit Intercept?",
                    description="Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.",
                    default="true",
                ),
                OptionsParameter[str](
                    name="multi_class",
                    display_name="Training Scheme",
                    description="Training algorithm scheme. If the option chosen is 'OvR', then a binary problem is fit for each label. For 'Multinomial' the loss minimized is the multinomial loss fit across the entire probability distribution, even when the data is binary. The cross-entropy loss is used for 'multinomial' in the multi-class case. 'Multinomial' is unavailable when the solver is 'liblinear'. 'Auto' selects 'OvR' if the data is binary, or if solver='liblinear', and otherwise selects 'multinomial'.",
                    default="auto",
                    is_multi=False,
                    options={
                        "auto": {"display_name": "Auto", "value": "auto"},
                        "ovr": {
                            "display_name": "One-vs-Rest (OvR)",
                            "value": "ovr",
                        },
                        "multinomial": {
                            "display_name": "Multinomial",
                            "value": "multinomial",
                        },
                    },
                    # NOTE: Currently the ‘multinomial’ option is supported only by the ‘lbfgs’, ‘sag’, ‘saga’ and ‘newton-cg’ solvers.
                ),
                # TODO: Include? Default is None
                # FloatParameter(
                # name="l1-ratio",
                #     display_name="L1 Ratio",
                #     description="The Elastic-Net mixing parameter. Only used if penalty='elasticnet'. Setting value to 0 is equivalent to using penalty='L2', while setting value to 1 is equivalent to using penalty='L1'. For values between 0 and 1, the penalty is a combination of L1 and L2.",
                #     default=1e-4,
                #     min=0,
                #     max=1.00001,
                # ),
            ]

        return Parameters(
            name=Model.lr,
            display_name="Linear Regression",
            parameters=parameters,
        )
