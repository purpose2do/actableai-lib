from functools import lru_cache

from actableai.clustering.models.base import ClusteringModelWrapperNoFit, Model
from actableai.parameters.numeric import IntegerParameter, FloatParameter
from actableai.parameters.options import OptionsParameter
from actableai.parameters.parameters import Parameters


# TODO: Add requirement for pyamg if use 'amg' for eigen_solver


class SpectralClustering(ClusteringModelWrapperNoFit):
    """Class to handle Spectral Clustering.

    Args:
        Base class for all clustering models
    """

    @staticmethod
    @lru_cache(maxsize=None)
    def get_parameters() -> Parameters:
        """Returns the parameters of the model.

        Returns:
            The parameters.
        """
        parameters = [
            IntegerParameter(
                name="random_state",
                display_name="Random State",
                description="Determines the random number generator.",
                default=0,
                # TODO check constraints
            ),
            IntegerParameter(
                name="n_init",
                display_name="KMeans Initialization Count",
                description="Number of time the k-means algorithm will be run with\
                different centroid seeds. The final results will be the best output\
                of n_init consecutive runs in terms of inertia. Only used if\
                assign_labels='KMeans'.",
                default=10,
                # TODO check constraints
            ),
            FloatParameter(
                name="gamma",
                display_name="Gamma",
                description="Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2\
                kernels. Ignored for affinity='nearest_neighbors'.",
                default=1.0,
                # TODO check constraints
            ),
            OptionsParameter[str](
                name="affinity",
                display_name="Affinity",
                description="How to construct the affinity matrix.",
                default="rbf",
                is_multi=False,
                options={
                    "rbf": {
                        "display_name": "Radial basis function",
                        "value": "rbf",
                    },
                    "nearest_neighbors": {
                        "display_name": "Nearest Neighbors",
                        "value": "nearest_neighbors",
                    },
                    "precomputed": {
                        "display_name": "Pre-computed",
                        "value": "precomputed",
                    },
                },
            ),
            IntegerParameter(
                name="n_neighbors",
                display_name="Neighbors Count",
                description="Number of neighbors to use when constructing the\
                affinity matrix using the nearest neighbors method. Ignored for\
                affinity='rbf'.",
                default=10,
                min=0,
                # TODO check constraints
            ),
            OptionsParameter[str](
                name="assign_labels",
                display_name="Labels Strategy",
                description="The strategy for assigning labels in the embedding space.",
                default="kmeans",
                is_multi=False,
                options={
                    "kmeans": {
                        "display_name": "KMeans",
                        "value": "kmeans",
                    },
                    "discretize": {
                        "display_name": "Discretize",
                        "value": "discretize",
                    },
                },
            ),
            FloatParameter(
                name="degree",
                display_name="Degree",
                description="Degree of the polynomial kernel. Ignored by other kernels.",
                default=3,
                # TODO check constraints
            ),
            FloatParameter(
                name="coef0",
                display_name="Zero Coefficient",
                description="Zero coefficient for polynomial and sigmoid kernels. Ignored by other kernels.",
                default=1,
                # TODO check constraints
            ),
        ]

        return Parameters(
            name=Model.spectral_clustering,
            display_name="Spectral Clustering",
            parameters=parameters,
        )

    def _initialize_model(self):
        from sklearn.cluster import SpectralClustering as _SpectralClustering

        
        self.model = _SpectralClustering(
            n_clusters=self.num_clusters,
            **self.parameters,
            n_jobs=1,
            verbose=self.verbosity > 0,
        )
