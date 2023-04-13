from functools import lru_cache

from actableai.clustering.models.base import ClusteringModelWrapperNoFit, Model
from actableai.parameters.options import OptionsParameter
from actableai.parameters.parameters import Parameters


class AgglomerativeClustering(ClusteringModelWrapperNoFit):
    """Class to handle Agglomerative Clustering.

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
            OptionsParameter[str](
                name="affinity",
                display_name="Metric",
                description="Metric used to compute the linkage.",
                default="euclidean",
                is_multi=False,
                options={
                    "euclidean": {
                        "display_name": "Euclidean",
                        "value": "euclidean",
                    },
                    "l1": {
                        "display_name": "L1",
                        "value": "l1",
                    },
                    "l2": {
                        "display_name": "L2",
                        "value": "l2",
                    },
                    "manhattan": {
                        "display_name": "Manhattan",
                        "value": "manhattan",
                    },
                    "cosine": {
                        "display_name": "Cosine",
                        "value": "cosine",
                    },
                },
            ),
            OptionsParameter[str](
                name="linkage",
                display_name="Linkage",
                description="Which linkage criterion to use. The linkage criterion\
                determines which distance to use between sets of observation. The\
                algorithm will merge the pairs of cluster that minimize this\
                criterion. 'Ward' minimizes the variance of the clusters being merged,\
                'complete' uses the maximum distances between all observations of the two\
                sets, 'average' uses the average of the distances of each observation of the\
                two sets, and 'single' uses the minimum of the distances between all\
                observations of the two sets",
                default="ward",
                is_multi=False,
                options={
                    "ward": {
                        "display_name": "Ward",
                        "value": "ward",
                    },
                    "complete": {
                        "display_name": "Complete",
                        "value": "complete",
                    },
                    "average": {
                        "display_name": "Average",
                        "value": "average",
                    },
                    "single": {
                        "display_name": "Single",
                        "value": "single",
                    },
                },
            ),
        ]

        return Parameters(
            name=Model.agglomerative_clustering,
            display_name="Agglomerative Clustering",
            parameters=parameters,
        )

    def _initialize_model(self):
        from sklearn.cluster import AgglomerativeClustering as _AgglomerativeClustering

        
        self.model = _AgglomerativeClustering(
            n_clusters=self.num_clusters,
            **self.parameters,
        )
