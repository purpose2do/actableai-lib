from functools import lru_cache

import numpy as np

from actableai.embedding.models.base import EmbeddingModelWrapper, Model
from actableai.parameters.numeric import IntegerParameter, FloatParameter
from actableai.parameters.options import OptionsParameter
from actableai.parameters.parameters import Parameters


class TSNE(EmbeddingModelWrapper):
    """Class to handle TSNE."""

    @staticmethod
    @lru_cache(maxsize=None)
    def get_parameters() -> Parameters:
        """Returns the parameters of the model.

        Returns:
            The parameters.
        """
        parameters = [
            FloatParameter(
                name="perplexity",
                display_name="Perplexity",
                description="The perplexity is related to the number of nearest\
                neighbors that is used in other manifold learning algorithms. Larger\
                datasets usually require a larger perplexity. Consider selecting a\
                value between 5 and 50. Different values can result in significantly\
                different results. The perplexity must be less than the number of\
                samples.",
                default=30,
                # TODO check constraints
            ),
            FloatParameter(
                name="early_exaggeration",
                display_name="Early Exaggeration",
                description="Controls how tight natural clusters in the\
                original space are in the embedded space and how much space will be\
                between them.",
                defeault=12,
                # TODO check constraints
            ),
            IntegerParameter(
                name="early_exaggeration_iter",
                display_name="Early Exaggeration Iterations",
                description="The number of iterations to run in the early exaggeration phase.",
                default=250,
                min=1,
                # TODO check constraints
            ),
            IntegerParameter(
                name="n_iter",
                display_name="Iterations Count",
                description="Maximum number of iterations for the optimization.",
                default=1000,
                min=250,
                # TODO check constraints
            ),
            OptionsParameter[str](
                name="metric",
                display_name="Metric",
                description="The metric to use when calculating distance between\
                instances in a feature array.",
                default="euclidean",
                is_multi=False,
                options={
                    "euclidean": {
                        "display_name": "Euclidian",
                        "value": "euclidean",
                    },
                    "manhattan": {
                        "display_name": "Manhattan",
                        "value": "manhattan",
                    },
                    "chebyshev": {
                        "display_name": "Chebyshev",
                        "value": "chebyshev",
                    },
                    "minkowski": {
                        "display_name": "Minkowski",
                        "value": "minkowski",
                    },
                    "canberra": {
                        "display_name": "Canberra",
                        "value": "canberra",
                    },
                    "braycurtis": {
                        "display_name": "Braycurtis",
                        "value": "braycurtis",
                    },
                    "mahalanobis": {
                        "display_name": "Mahalanobis",
                        "value": "mahalanobis",
                    },
                    "dice": {
                        "display_name": "Dice",
                        "value": "dice",
                    },
                    # TODO add more losses, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
                },
            ),
            OptionsParameter[str](
                name="initialization",
                display_name="Initialization",
                description="The metric to use when calculating distance between instances in a feature array.",
                default="pca",
                is_multi=False,
                options={
                    "spectral": {
                        "display_name": "Spectral",
                        "value": "spectral",
                    },
                    "pca": {
                        "display_name": "Principal Component Analysis",
                        "value": "pca",
                    },
                    "random": {
                        "display_name": "Random",
                        "value": "random",
                    },
                },
            ),
            IntegerParameter(
                name="random_state",
                display_name="Random State",
                description="Determines the random number generator.",
                default=0,
            ),
            OptionsParameter[str](
                name="negative_gradient_method",
                display_name="Negative Gradient Method",
                description="Specifies the negative gradient approximation method to use.\
                For smaller data sets, the Barnes-Hut approximation is appropriate.\
                For larger data sets, the FFT accelerated interpolation\
                method is more appropriate.\
                Alternatively, you can use auto to approximately select the faster method.",
                default="bh",
                is_multi=False,
                options={
                    "bh": {
                        "display_name": "Barnes-Hut",
                        "value": "bh",
                    },
                    "fft": {
                        "display_name": "FFT",
                        "value": "fft",
                    },
                    "auto": {
                        "display_name": "Auto",
                        "value": "auto",
                    },
                },
            ),
            FloatParameter(
                name="theta",
                display_name="Theta",
                description="Only used if method='barnes_hut' This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.",
                default=0.5,
                # TODO check constraints
            ),
        ]

        return Parameters(
            name=Model.tsne,
            display_name="t-SNE",
            parameters=parameters,
        )

    def _initialize_model(self):
        from openTSNE import TSNE as _TSNE

        
        self.model = _TSNE(
            n_components=self.embedding_size,
            **self.parameters,
            verbose=self.verbosity > 0,
            n_jobs=1,
        )

    def _fit_transform(self, data: np.ndarray, target: np.ndarray = None) -> np.ndarray:
        self.fit(data)
        return np.array(self.model)
