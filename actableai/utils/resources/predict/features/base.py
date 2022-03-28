from abc import ABC, abstractmethod
from typing import Tuple

from actableai.utils.resources.predict import ResourcePredictorType


class FeaturesExtractor(ABC):
    """
    Base class for features extractors
    """

    def __init__(self, resource_predicted: ResourcePredictorType):
        """
        MethodFeaturesExtractor constructor

        Parameters
        ----------
        resource_predicted:
            The resource to predict
        """
        self.resource_predicted = resource_predicted

    def _filter_features(self, features: dict) -> dict:
        """
        Abstract method to filter features, if not implemented no filtering will be done

        Parameters
        ----------
        features:
            The features to filter

        Returns
        -------
        The filtered features
        """
        return features

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Tuple[dict, dict]:
        """
        Abstract method which is called to extract features

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
        raise NotImplementedError
