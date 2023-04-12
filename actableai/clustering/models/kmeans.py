from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans as _KMeans
from sklearn.metrics import silhouette_score

from actableai.clustering.models.base import ClusteringModelWrapper, Model
from actableai.parameters.numeric import (
    FloatParameter,
    IntegerParameter,
)
from actableai.parameters.options import OptionsParameter
from actableai.parameters.parameters import Parameters


class KMeans(ClusteringModelWrapper):
    """Class to handle K-Means clustering.

    Args:
        Base class for all clustering models
    """

    @staticmethod
    def KMeans_scaled_inertia(
        scaled_data: np.ndarray, k: int, alpha_k: float, *KMeans_args, **KMeans_kwargs
    ):
        """KMeans with scaled inertia.

        Args:
            scaled_data: matrix scaled data. rows are samples and columns are features for
                clustering.
            k: current k for applying KMeans.
            alpha_k: manually tuned factor that gives penalty to the number of clusters.

        Returns:
            float: scaled inertia value for current k
        """

        # fit k-means
        inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
        kmeans = _KMeans(n_clusters=k, *KMeans_args, **KMeans_kwargs).fit(scaled_data)
        scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
        return scaled_inertia

    @classmethod
    def KMeans_pick_k(
        cls, scaled_data, alpha_k, k_range, *KMeans_args, **KMeans_kwargs
    ) -> _KMeans:
        """Find best k for KMeans based on scaled inertia method.
            https://towardsdatascience.com/an-approach-for-choosing-number-of-clusters-for-k-means-c28e614ecb2c

        Args:
            scaled_data: matrix scaled data. rows are samples and columns are features for
                clustering.
            alpha_k: manually tuned factor that gives penalty to the number of clusters.
            k_range: range of k values to test.

        Returns:
            best_k: The value of the best k.
        """
        ans = []
        for k in k_range:
            scaled_inertia = cls.KMeans_scaled_inertia(
                scaled_data, k, alpha_k, *KMeans_args, **KMeans_kwargs
            )
            ans.append((k, scaled_inertia))
        results = pd.DataFrame(ans, columns=["k", "Scaled Inertia"]).set_index("k")
        best_k = results.idxmin()[0]
        return best_k

    @staticmethod
    def KMeans_pick_k_sil(X, k_range, *KMeans_args, **KMeans_kwargs):
        """Find best k for KMeans based on silhouette score.
            https://newbedev.com/scikit-learn-k-means-elbow-criterion

        Args:
            X: matrix of data. rows are samples and columns are features for
                clustering.
            k_range: range of k values to test.

        Returns:
            best_k: The value of the best k.
        """
        max_sil_coeff, best_k = 0, 2
        for k in k_range:
            kmeans = _KMeans(n_clusters=k).fit(X)
            label = kmeans.labels_
            sil_coeff = silhouette_score(X, label, metric="euclidean")
            print("Cluster: ", k, ", Silhouette coeff: ", sil_coeff)
            if max_sil_coeff < sil_coeff:
                max_sil_coeff = sil_coeff
                best_k = k
        return best_k

    @classmethod
    def find_num_clusters(
        cls,
        data: np.ndarray,
        k_select_method: str,
        auto_num_clusters_min: int,
        auto_num_clusters_max: int,
        alpha_k: float = 0.01,
    ) -> int:
        num_clusters = None
        if k_select_method == "silhouette":
            num_clusters = cls.KMeans_pick_k_sil(
                data,
                range(auto_num_clusters_min, auto_num_clusters_max + 1),
            )

        elif k_select_method == "scaled_inertia":
            num_clusters = cls.KMeans_pick_k(
                data,
                alpha_k,
                range(auto_num_clusters_min, auto_num_clusters_max + 1),
            )
        else:
            ValueError(f"Unknown select method: {k_select_method}")

        return num_clusters

    @staticmethod
    @lru_cache(maxsize=None)
    def get_parameters() -> Parameters:
        """Returns the parameters of the model.

        Returns:
            The parameters.
        """

        parameters = [
            OptionsParameter[str](
                name="init",
                display_name="Initialization Method",
                description="Method to use to initialize the cluster centers.",
                default="k-means++",
                is_multi=False,
                options={
                    "k-means++": {
                        "display_name": "k-means++",
                        "value": "k-means++",
                    },
                    "random": {
                        "display_name": "Random",
                        "value": "random",
                    },
                }
                # TODO: Check if should add ability to use array
            ),
            IntegerParameter(
                name="n_init",
                display_name="Initializations Count",
                description="Number of times the k-means algorithm will be run with different centroid seeds.\
                The final results will be the best output of n_init consecutive runs in terms of inertia.",
                default=10,
                min=1,
                # TODO check constraints
            ),
            IntegerParameter(
                name="max_iter",
                display_name="Maximum Iterations",
                description="Maximum number of iterations of the k-means algorithm for a single run.",
                default=300,
                min=1,
                # TODO check constraints
            ),
            FloatParameter(
                name="tol",
                display_name="Tolerance",
                description="Relative tolerance with regards to Frobenius norm of the\
                difference in the cluster centers of two consecutive iterations to\
                declare convergence.",
                default=1e-4,
                min=0,
                # TODO check constraints
            ),
            IntegerParameter(
                name="random_state",
                display_name="Random State",
                description="Pseudo-random number generator to control the starting\
                state, enabling reproducible results across function calls.",
                default=0,
            ),
            OptionsParameter[str](
                name="algorithm",
                display_name="Algorithm",
                description="K-Means algorithm to use.",
                default="auto",
                is_multi=False,
                options={
                    "auto": {
                        "display_name": "Automatic",
                        "value": "auto",
                    },
                    "lloyd": {
                        "display_name": "lloyd",
                        "value": "lloyd",
                    },
                    "elkan": {
                        "display_name": "Elkan",
                        "value": "elkan",
                    },
                },
            ),
        ]

        return Parameters(
            name=Model.kmeans,
            display_name="K-Means",
            parameters=parameters,
        )

    def _initialize_model(self):
        self.model = _KMeans(
            n_clusters=self.num_clusters,
            **self.parameters,
            verbose=self.verbosity > 0,
        )
