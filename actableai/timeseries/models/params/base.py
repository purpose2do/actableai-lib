from enum import Enum, unique
from functools import lru_cache
from typing import Callable, Any, Dict, Union, Tuple, Optional

from gluonts.model.estimator import Estimator
from gluonts.model.predictor import Predictor
from gluonts.mx.distribution import DistributionOutput
from hyperopt import hp
from mxnet.context import Context

from actableai.parameters.numeric import FloatRangeSpace, IntegerRangeSpace
from actableai.parameters.parameters import Parameters
from actableai.parameters.type import ParameterType, ValueType
from actableai.timeseries.models.estimator import AAITimeSeriesEstimator
from actableai.timeseries.models.predictor import AAITimeSeriesPredictor
from actableai.timeseries.transform.base import Transformation
from actableai.timeseries.transform.clean_features import CleanFeatures
from actableai.timeseries.transform.identity import Identity


class BaseParams:
    """Base class for Time Series Model Parameters."""

    def __init__(
        self,
        model_name: str,
        is_multivariate_model: bool,
        has_estimator: bool = True,
        handle_feat_static_real: bool = True,
        handle_feat_static_cat: bool = True,
        handle_feat_dynamic_real: bool = False,
        handle_feat_dynamic_cat: bool = False,
        hyperparameters: Dict = None,
        process_hyperparameters: bool = True,
    ):
        """BaseParams Constructor.

        Args:
            model_name: Name of the model.
            is_multivariate_model: Flag to indicate whether the underlying model is a
                multivariate model or not.
            has_estimator: Flag to indicate whether the `build_estimator` function is
                implemented or not.
            handle_feat_static_real: Whether the underlying model is handling static
                real features or not.
            handle_feat_static_cat: Whether the underlying model is handling static cat
                features or not.
            handle_feat_dynamic_real: Whether the underlying model is handling dynamic
                real features or not.
            handle_feat_dynamic_cat: Whether the underlying model is handling dynamic
                cat features or not.
            hyperparameters: Dictionary representing the hyperparameters space.
            process_hyperparameters: If True the hyperparameters will be validated and
                processed (deactivate if they have already been validated).
        """
        self.hyperparameters = hyperparameters
        if self.hyperparameters is None:
            self.hyperparameters = {}

        if process_hyperparameters:
            hyperparameters_space = self.get_hyperparameters()

            (
                hyperparameters_validation,
                self.hyperparameters,
            ) = hyperparameters_space.validate_process_parameter(self.hyperparameters)

            if len(hyperparameters_validation) > 0:
                raise ValueError(str(hyperparameters_validation))

        self.model_name = model_name
        self.is_multivariate_model = is_multivariate_model
        self.has_estimator = has_estimator

        self.handle_feat_static_real = handle_feat_static_real
        self.handle_feat_static_cat = handle_feat_static_cat
        self.handle_feat_dynamic_real = handle_feat_dynamic_real
        self.handle_feat_dynamic_cat = handle_feat_dynamic_cat

        self.use_feat_static_real = False
        self.use_feat_static_cat = False
        self.use_feat_dynamic_real = False
        self.use_feat_dynamic_cat = False

        self._transformation = Identity()

    def _hp_param(self, func: Callable, param_name: str, *args, **kwargs) -> Any:
        """Util function used to call hyperopt parameter function.

        Args:
            func: Hyperopt function to call.
            param_name: Name of the parameter.
            args: Arguments to pass to the function.
            kwargs: Named arguments to pass to the function.

        Returns:
            Result of the function.
        """
        return func(f"{self.model_name}_{param_name}", *args, **kwargs)

    def _auto_select(self, param_name: str) -> Any:
        """Util function used to automatically call the right hyperparameter selection.

        Args:
            param_name: Name of the parameter.

        Returns:
            Choose parameter value.
        """
        parameter = self.get_hyperparameters().parameters[param_name]
        parameter_type = parameter.parameter_type

        options = self.hyperparameters[param_name]

        if parameter_type == ParameterType.VALUE or (
            parameter_type == ParameterType.LIST
            and isinstance(parameter, (FloatRangeSpace, IntegerRangeSpace))
        ):
            value_type = parameter.value_type
            if value_type == ValueType.INT:
                return self._randint(param_name, options)
            if value_type == ValueType.FLOAT:
                return self._uniform(param_name, options)
            if value_type == ValueType.BOOL:
                return options
        if (
            parameter_type == ParameterType.OPTIONS
            or parameter_type == ParameterType.PARAMETERS
            or parameter_type == ParameterType.LIST
        ):
            return self._choice(param_name, options)

        raise ValueError("Invalid parameter type")

    def _choice(self, param_name: str, options: Union[Tuple[Any, ...], Any]) -> Any:
        """Util function to represent a parameter selection over a list.

        Args:
            param_name: Name of the parameter.
            options: List to choose from.

        Returns:
            Choose parameter value.
        """
        if not isinstance(options, (list, tuple)):
            return options
        return self._hp_param(hp.choice, param_name, options)

    def _randint(self, param_name: str, options: Union[Tuple[int, int], int]) -> int:
        """Util function to represent a parameter selection of an integer.

        Args:
            param_name: Name of the parameter.
            options: Tuple representing the minimum and maximum (excluded) integer.

        Returns:
            Choose parameter value.
        """
        if not isinstance(options, (list, tuple)):
            return options
        return self._hp_param(hp.randint, param_name, *options)

    def _uniform(
        self, param_name: str, options: Union[Tuple[float, float], float]
    ) -> float:
        """Util function to represent a parameter selection of a floating point.

        Args:
            param_name: Name of the parameter.
            options: Tuple representing the minimum and maximum (excluded) float.

        Returns:
            Choose parameter value.
        """
        if not isinstance(options, (list, tuple)):
            return options
        return self._hp_param(hp.uniform, param_name, *options)

    def _get_context_length(self, prediction_length: int) -> Optional[int]:
        """Util function to compute the context length using the prediction length and
            the context length ratio.

        Args:
            prediction_length: Length of the prediction.

        Returns:
             Computed context length.
        """
        if "context_length_ratio" not in self.hyperparameters:
            return None

        context_length = self.hyperparameters["context_length_ratio"]
        if isinstance(context_length, (tuple, list)):
            context_length = tuple(
                [round(e * prediction_length) for e in context_length]
            )
        else:
            context_length = round(context_length * prediction_length)

        return context_length

    def setup(
        self,
        use_feat_static_real: bool,
        use_feat_static_cat: bool,
        use_feat_dynamic_real: bool,
        use_feat_dynamic_cat: bool,
    ):
        """Set up the parameters.

        Args:
            use_feat_static_real: True if the data contains real static features.
            use_feat_static_cat: True if the data contains categorical static features.
            use_feat_dynamic_real: True if the data contains real dynamic features.
            use_feat_dynamic_cat: True if the data contains categorical dynamic
                features.
        """
        self.use_feat_static_real = (
            self.handle_feat_static_real and use_feat_static_real
        )
        self.use_feat_static_cat = self.handle_feat_static_cat and use_feat_static_cat
        self.use_feat_dynamic_real = (
            self.handle_feat_dynamic_real and use_feat_dynamic_real
        )
        self.use_feat_dynamic_cat = (
            self.handle_feat_dynamic_cat and use_feat_dynamic_cat
        )

        self._transformation += CleanFeatures(
            keep_feat_static_real=self.use_feat_static_real,
            keep_feat_static_cat=self.use_feat_static_cat,
            keep_feat_dynamic_real=self.use_feat_dynamic_real,
            keep_feat_dynamic_cat=self.use_feat_dynamic_cat,
        )

    def tune_config(self, prediction_length: int) -> Dict[str, Any]:
        """Select parameters in the pre-defined hyperparameter space.

        Args:
            prediction_length: Length of the prediction.

        Returns:
            Selected parameters.
        """
        return {}

    def build_estimator(
        self,
        *,
        ctx: Context,
        freq: str,
        prediction_length: int,
        target_dim: int,
        distr_output: DistributionOutput,
        params: Dict[str, Any],
    ) -> Optional[AAITimeSeriesEstimator]:
        """Build an estimator from the underlying model using selected parameters.

        Args:
            ctx: mxnet context.
            freq: Frequency of the time series used.
            prediction_length: Length of the prediction that will be forecasted.
            target_dim: Target dimension (number of columns to predict).
            distr_output: Distribution output to use.
            params: Selected parameters from the hyperparameter space.

        Returns:
            Built estimator.
        """
        return None

    def build_predictor(
        self,
        *,
        freq: str,
        prediction_length: int,
        target_dim: int,
        params: Dict[str, Any],
    ) -> Optional[AAITimeSeriesPredictor]:
        """Build a predictor from the underlying model using selected parameters.

        Args:
            freq: Frequency of the time series used.
            prediction_length: Length of the prediction that will be forecasted.
            target_dim: Target dimension (number of columns to predict).
            params: Selected parameters from the hyperparameter space.

        Returns:
            Built predictor.
        """
        return None

    def _create_estimator(
        self,
        estimator: Estimator,
        additional_transformation: Optional[Transformation] = None,
    ) -> AAITimeSeriesEstimator:
        """Create the estimator associated with the model.

        Args:
            estimator: Underlying GluonTS estimator.
            additional_transformation: Additional transformation to add to this specific
                estimator. Will be chained with the BaseParams transformation.

        Returns:
            The wrapped estimator.
        """
        return AAITimeSeriesEstimator(
            estimator=estimator,
            transformation=(self._transformation + additional_transformation),
        )

    def _create_predictor(
        self,
        predictor: Predictor,
        additional_transformation: Optional[Transformation] = None,
    ) -> AAITimeSeriesPredictor:
        """Create the predictor associated with the model.

        Args:
            predictor: Underlying GluonTS predictor.
            additional_transformation: Additional transformation to add to this specific
                estimator. Will be chained with the BaseParams transformation.

        Returns:
            The wrapped predictor.
        """
        return AAITimeSeriesPredictor(
            predictor=predictor,
            transformation=(self._transformation + additional_transformation),
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def get_hyperparameters() -> Parameters:
        """Returns the hyperparameters space of the model.

        Returns:
            The hyperparameters space.
        """
        raise NotImplementedError


@unique
class Model(str, Enum):
    """Enum representing the different model available."""

    constant_value = "constant_value"
    multivariate_constant_value = "multivariate_constant_value"
    deep_ar = "deep_ar"
    deep_var = "deep_var"
    feed_forward = "feed_forward"
    gp_var = "gp_var"
    n_beats = "n_beats"
    prophet = "prophet"
    r_forecast = "r_forecast"
    tree_predictor = "tree_predictor"
