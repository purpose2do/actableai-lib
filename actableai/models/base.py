from __future__ import annotations


from abc import ABC, abstractmethod
import logging
from functools import lru_cache
from typing import Dict, Any

import numpy as np

from actableai.parameters.parameters import Parameters


class AAIBaseModel(ABC):
    has_fit: bool = True
    has_transform: bool = True
    has_predict: bool = True

    def __init__(self):
        self.is_fit = False

    def fit_transform(self, data: np.ndarray, target: np.ndarray = None) -> np.ndarray:
        if not self.has_fit or not self.has_transform:
            logging.warning("Calling fit_transform has no effect.")
            return data

        return self._fit_transform(data, target)

    def _fit_transform(self, data: np.ndarray, target: np.ndarray = None) -> np.ndarray:
        self.fit(data, target)
        return self.transform(data)

    def fit_predict(self, data: np.ndarray, target: np.ndarray = None) -> np.ndarray:
        if not self.has_fit or not self.has_predict:
            logging.warning("Calling fit_predict has no effect.")
            return target

        return self._fit_predict(data, target)

    def _fit_predict(self, data: np.ndarray, target: np.ndarray = None) -> np.ndarray:
        self.fit(data, target)
        return self.predict(data)

    def fit(self, data: np.ndarray, target: np.ndarray = None) -> "AAIBaseModel":
        if not self.has_fit:
            logging.warning("Calling fit has no effect.")
            return self

        if self.is_fit:
            raise RuntimeError("Model already fitted.")

        self.is_fit = self._fit(data, target)

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.has_transform:
            logging.warning("Calling transform has no effect.")
            return data

        if self.has_fit and not self.is_fit:
            raise RuntimeError("Model not fitted.")

        return self._transform(data)

    def predict(self, data: np.ndarray) -> np.ndarray:
        if not self.has_predict:
            logging.warning("Calling predict has no effect.")
            return None

        if self.has_fit and not self.is_fit:
            raise RuntimeError("Model not fitted.")

        return self._predict(data)

    # Functions that need to be overriden
    def _fit(self, data: np.ndarray, target: np.ndarray = None) -> bool:
        raise NotImplementedError()

    def _transform(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def _predict(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class AAIParametersModel(AAIBaseModel, ABC):
    @staticmethod
    @abstractmethod
    @lru_cache(maxsize=None)
    def get_parameters() -> Parameters:
        """Returns the parameters of the model.

        Returns:
            The parameters.
        """
        raise NotImplementedError()

    def __init__(
        self,
        parameters: Dict[str, Any] = None,
        process_parameters: bool = True,
    ):
        super().__init__()

        self.parameters = parameters
        if self.parameters is None:
            self.parameters = {}

        # Process parameters if needed
        if process_parameters:
            parameters_definition = self.get_parameters()

            (
                parameters_validation,
                self.parameters,
            ) = parameters_definition.validate_process_parameter(self.parameters)

            if len(parameters_validation) > 0:
                raise ValueError(str(parameters_validation))
