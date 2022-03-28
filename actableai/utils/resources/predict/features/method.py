import inspect
from abc import abstractmethod
from typing import Callable, OrderedDict, Tuple

from actableai.utils.resources.predict import ResourcePredictorType
from actableai.utils.resources.predict.features.base import FeaturesExtractor


class MethodFeaturesExtractor(FeaturesExtractor):
    """
    Base class for method features extractors
    """

    def __init__(self, resource_predicted: ResourcePredictorType, function: Callable):
        """
        MethodFeaturesExtractor constructor

        Parameters
        ----------
        resource_predicted:
            The resource to predict

        function:
            The method to extract the features from
        """
        super().__init__(resource_predicted)
        self.signature = inspect.signature(function)

    def _bind_arguments(self, *args, **kwargs) -> OrderedDict:
        """
        Get all the arguments to extract the features from (with all the default values)

        Parameters
        ----------
        args:
            The arguments passed to the function
        kwargs:
            The named arguments passed to the function

        Returns
        -------
        The arguments dictionary
        """
        arguments = self.signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        return arguments.arguments

    @staticmethod
    @abstractmethod
    def _extract_all_features(arguments: dict) -> dict:
        """
        Extract all features, must be implemented

        Parameters
        ----------
        arguments:
            The arguments used to call the method and to extract the features from

        Returns
        -------
        The extracted features
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tuple[dict, dict]:
        """
        Method used to extract features

        Parameters
        ----------
        args:
            The arguments passed to the function
        kwargs:
            The named arguments passed to the function

        Returns
        -------
        The extracted features, return an empty dict if the resource to predicted is not implemented for this task
        """
        arguments = self._bind_arguments(*args, **kwargs)

        # Remove object
        arguments.pop("self", None)
        arguments.pop("cls", None)

        features = self._extract_all_features(arguments)

        return self._filter_features(features), features
