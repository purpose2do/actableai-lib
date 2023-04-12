import logging
from abc import abstractmethod, ABC
from enum import unique, Enum
from functools import lru_cache
from typing import Dict, Any

import numpy as np

from actableai.models.base import AAIParametersModel
from actableai.parameters.parameters import Parameters


class BaseClusteringModel(AAIParametersModel, ABC):
    """
    TODO write documentation
    """

    has_fit = True
    has_transform = False
    has_predict = True
    has_explanations = False
    handle_categorical = False

    @staticmethod
    @abstractmethod
    @lru_cache(maxsize=None)
    def get_parameters() -> Parameters:
        raise NotImplementedError()

    def __init__(
        self,
        input_size: int,
        num_clusters: int,
        parameters: Dict[str, Any] = None,
        process_parameters: bool = True,
        verbosity: int = 1,
    ):
        super().__init__(
            parameters=parameters,
            process_parameters=process_parameters,
        )

        self.input_size = input_size
        self.num_clusters = num_clusters
        self.verbosity = verbosity

    def project(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError("Model not fitted")

        return self._project(data)

    def _project(self, data: np.ndarray) -> np.ndarray:
        logging.warning(
            "No specific project function for this model, returning input data."
        )
        return data

    def explain_samples(self, data: np.ndarray) -> np.ndarray:
        if not self.has_explanations:
            logging.warning("No explanations available")
            return data

        return self._explain_samples(data)

    def _explain_samples(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class ClusteringModelWrapper(BaseClusteringModel, ABC):
    def __init__(
        self,
        input_size: int,
        num_clusters: int,
        parameters: Dict[str, Any] = None,
        process_parameters: bool = True,
        verbosity: int = 1,
    ):
        super().__init__(
            input_size=input_size,
            num_clusters=num_clusters,
            parameters=parameters,
            process_parameters=process_parameters,
            verbosity=verbosity,
        )

        self.model = None
        self.initialize_model()

    def initialize_model(self):
        self.is_fit = False
        self._initialize_model()

    @abstractmethod
    def _initialize_model(self):
        raise NotImplementedError()

    def _fit(self, data: np.ndarray, target: np.ndarray = None) -> bool:
        self.model = self.model.fit(data)
        return True

    def _predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict(data)


class ClusteringModelWrapperNoFit(ClusteringModelWrapper, ABC):
    def _fit(self, data: np.ndarray, target: np.ndarray = None) -> bool:
        return True

    def _predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(data)


@unique
class Model(str, Enum):
    """Enum representing the different model available."""

    dec = "dec"
    affinity_propagation = "affinity_propagation"
    agglomerative_clustering = "agglomerative_clustering"
    dbscan = "dbscan"
    kmeans = "kmeans"
    spectral_clustering = "spectral_clustering"
