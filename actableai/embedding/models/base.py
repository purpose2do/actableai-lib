from abc import ABC, abstractmethod
from enum import unique, Enum
from typing import Dict, Any

import numpy as np

from actableai.models.base import AAIParametersModel


class BaseEmbeddingModel(AAIParametersModel, ABC):
    has_fit = True
    has_transform = True
    has_predict = False

    def __init__(
        self,
        embedding_size: int = 2,
        parameters: Dict[str, Any] = None,
        process_parameters: bool = True,
        verbosity: int = 1,
    ):
        super().__init__(
            parameters=parameters,
            process_parameters=process_parameters,
        )

        self.embedding_size = embedding_size
        self.verbosity = verbosity


class EmbeddingModelWrapper(BaseEmbeddingModel, ABC):
    def __init__(
        self,
        embedding_size: int = 2,
        parameters: Dict[str, Any] = None,
        process_parameters: bool = True,
        verbosity: int = 1,
    ):
        super().__init__(
            embedding_size=embedding_size,
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

    def _transform(self, data: np.ndarray) -> np.ndarray:
        return self.model.transform(data)

    def _predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict(data)


@unique
class Model(str, Enum):
    """Enum representing the different model available."""

    tsne = "tsne"
    umap = "umap"
    linear_discriminant_analysis = "linear_discriminant_analysis"
