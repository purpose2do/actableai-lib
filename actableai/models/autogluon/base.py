from __future__ import annotations

from typing import Any, Union, Tuple, Literal, TYPE_CHECKING, Type, Dict

from actableai.parameters.base import BaseParameter
from actableai.parameters.numeric import FloatRangeSpace, IntegerRangeSpace

if TYPE_CHECKING:
    from autogluon.core.space import Int, Real, Categorical
    from autogluon.core.models import AbstractModel

from abc import ABC
from enum import Enum, unique
from functools import lru_cache

from actableai.parameters.parameters import Parameters


class BaseParams(ABC):
    """Base class for Regression Model Parameters."""

    supported_problem_types: Literal["regression", "binary", "multiclass", "quantile"]
    _autogluon_name: str
    explain_samples_supported: bool
    gpu_required: bool = False

    @classmethod
    def get_autogluon_name(cls) -> Union[str, Type[AbstractModel]]:
        return cls._autogluon_name

    @classmethod
    def _auto_select(
        cls,
        options: Union[Tuple[Any, ...], Any],
        parameter_space: BaseParameter,
    ) -> Any:
        """Util function used to automatically call the right hyperparameter selection.

        Args:
            param_name: Name of the parameter.
            options: The options of the parameter. Can be a list, Tuple
                representing the minimum and maximum (excluded) integer/float, or a
                single value.
            parameter_space: The parameter space of a model.

        Returns:
            Chosen parameter value.
        """
        from actableai.parameters.numeric import FloatRangeSpace, IntegerRangeSpace
        from actableai.parameters.boolean import BooleanSpace
        from actableai.parameters.type import ParameterType, ValueType

        parameter_type = parameter_space.parameter_type

        if parameter_type == ParameterType.VALUE or (
            parameter_type == ParameterType.LIST
            and isinstance(parameter_space, (FloatRangeSpace, IntegerRangeSpace))
        ):
            value_type = parameter_space.value_type
            if value_type == ValueType.INT:
                return cls._randint(options)
            if value_type == ValueType.FLOAT:
                return cls._uniform(options, parameter_space)
            if value_type == ValueType.BOOL or value_type == ValueType.STR:
                return options

        if isinstance(parameter_space, BooleanSpace):
            return cls._bool(options)

        if (
            parameter_type == ParameterType.OPTIONS
            or parameter_type == ParameterType.PARAMETERS
            or parameter_type == ParameterType.LIST
        ):
            return cls._choice(options)

        raise ValueError("Invalid parameter type")

    @staticmethod
    def _choice(options: Union[Tuple[Any, ...], Any]) -> Any:
        """Util function to represent a parameter selection over a list.

        Args:
            options: List to choose from.

        Returns:
            Chosen parameter value.
        """
        import autogluon.core as ag

        if not isinstance(options, (list, tuple)):
            return options

        return ag.space.Categorical(
            *options
        )  # NOTE: First value assumed to be default (used first in HPO)

    @staticmethod
    def _randint(options: Union[Tuple[int, int], int]) -> Union[int, Int]:
        """Util function to represent a parameter selection of an integer.

        Args:
            options: Tuple representing the minimum and maximum (excluded) integer.

        Returns:
            Chosen parameter value.
        """
        import autogluon.core as ag

        if not isinstance(options, (list, tuple)):
            return options
        return ag.space.Int(lower=options[0], upper=options[-1])

    @staticmethod
    def _uniform(
        options: Union[Tuple[float, float], float],
        parameter: Union[FloatRangeSpace, IntegerRangeSpace],
    ) -> Union[float, Real]:
        """Util function to represent a parameter selection of a floating point.

        Args:
            options: Tuple representing the minimum and maximum (excluded) float.
            parameter: The parameter considered

        Returns:
            Chosen parameter value.
        """
        import autogluon.core as ag

        # Single value (parameter)
        if not isinstance(options, (list, tuple)):
            return options

        # Range Space
        log = parameter.is_log()
        return ag.space.Real(lower=options[0], upper=options[-1], log=log)

    @staticmethod
    def _bool(options: Union[Tuple[bool, bool], bool]) -> Union[bool, Categorical]:
        """Util function to represent a parameter selection of a boolean.

        Args:
            options: Boolean space

        Returns:
            Chosen parameter value.
        """
        import autogluon.core as ag

        if not isinstance(options, (list, tuple)):
            return options
        # TODO: Consider using ag.space.Bool(), but it may sometimes fail:
        return ag.space.Categorical(True, False)

    @classmethod
    def get_autogluon_parameters(
        cls,
        hyperparameters: Dict[str, Any],
        model_hyperparameters_space: Parameters,
        process_hyperparameters: bool = True,
    ) -> dict:
        """Converts hyperparameters for use in AutoGluon's hyperparameter
        optimization search.

        Args:
            params: The hyperparameters to be used by the model
            model_hyperparameters_space: The model's hyperparameter space
            process_hyperparameters: If True the hyperparameters will be validated and
                processed (deactivate if they have already been validated).

        Returns:
            parameters_autogluon: dictionary with hyperparameters in AutoGluon
                format.
        """

        if process_hyperparameters:
            (
                hyperparameters_validation,
                hyperparameters,
            ) = model_hyperparameters_space.validate_process_parameter(hyperparameters)

            if len(hyperparameters_validation) > 0:
                raise ValueError(str(hyperparameters_validation))

        # For each parameter in the model, convert the feature to AutoGluon format
        parameters_autogluon = dict()
        for param_name in hyperparameters:
            options = hyperparameters[param_name]
            parameter_space = model_hyperparameters_space.parameters[param_name]

            parameters_selected = cls._auto_select(
                options=options,
                parameter_space=parameter_space,
            )

            parameters_autogluon[param_name] = parameters_selected

        return parameters_autogluon

    @classmethod
    @lru_cache(maxsize=None)
    def get_hyperparameters(
        cls, *, problem_type: str, device: str, num_class: int
    ) -> Parameters:
        """Returns the hyperparameters space of the model.

        Args:
            problem_type: Defines the type of the problem (e.g. regression,
                binary classification, etc.).
            device: Which device is being used, can be one of 'cpu' or 'gpu'
            num_class: The number of classes, used for multi-class
                classification.

        Returns:
            The hyperparameters space.
        """
        if problem_type not in cls.supported_problem_types:
            raise ValueError(
                f"'{problem_type}' not supported for '{cls.__name__}'! 'problem_type' must be one of: {cls.supported_problem_types}"
            )

        return cls._get_hyperparameters(
            problem_type=problem_type,
            device=device,
            num_class=num_class,
        )

    @classmethod
    def _get_hyperparameters(
        cls, *, problem_type: str, device: str, num_class: int
    ) -> Parameters:
        raise NotImplementedError()


@unique
class Model(str, Enum):
    """Enum representing the different models available."""

    gbm = "gbm"
    cat = "cat"
    xgb_tree = "xgb_tree"
    xgb_linear = "xgb_linear"
    rf = "rf"
    xt = "xt"
    knn = "knn"
    lr = "lr"
    nn_torch = "nn_torch"
    nn_mxnet = "nn_mxnet"
    nn_fastainn = "nn_fastainn"
    ag_automm = "ag_automm"
    fasttext = "fasttext"
    tabpfn = "tabpfn"
