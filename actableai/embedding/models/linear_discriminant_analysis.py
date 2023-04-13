from functools import lru_cache

import numpy as np

from actableai.embedding.models.base import EmbeddingModelWrapper, Model
from actableai.parameters.options import OptionsParameter
from actableai.parameters.parameters import Parameters


class LinearDiscriminantAnalysis(EmbeddingModelWrapper):
    """Class to handle LDA."""

    @staticmethod
    @lru_cache(maxsize=None)
    def get_parameters() -> Parameters:
        """Returns the parameters of the model.

        Returns:
            The parameters.
        """
        parameters = [
            OptionsParameter[str](
                name="solver",
                display_name="Solver",
                description="Solver to use.",
                default="svd",
                is_multi=False,
                options={
                    "svd": {
                        "display_name": "Singular Value Decomposition",
                        "value": "svd",
                    },
                    "eigen": {
                        "display_name": "Eigenvalue Decomposition",
                        "value": "eigen",
                    },
                },
            ),
        ]

        return Parameters(
            name=Model.linear_discriminant_analysis,
            display_name="Linear Discriminant Analysis",
            parameters=parameters,
        )

    def _initialize_model(self):
        from sklearn.discriminant_analysis import (
            LinearDiscriminantAnalysis as _LinearDiscriminantAnalysis,
        )

        
        self.model = _LinearDiscriminantAnalysis(
            n_components=self.embedding_size,
            **self.parameters,
        )

    def _fit(self, data: np.ndarray, target: np.ndarray = None) -> bool:
        noise = np.random.normal(0, 0.000001, data.shape)
        data = data.astype(np.float64) + noise

        self.model.fit(data, target)
        return True
