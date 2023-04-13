from functools import lru_cache

from actableai.clustering.models.base import ClusteringModelWrapperNoFit, Model
from actableai.parameters.numeric import (
    FloatParameter,
    IntegerParameter,
)
from actableai.parameters.options import OptionsParameter
from actableai.parameters.parameters import Parameters


class DBSCAN(ClusteringModelWrapperNoFit):
    """Class to handle DBSCAN.

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
            FloatParameter(
                name="eps",
                display_name="Maximum Samples Distance",
                description="The maximum distance between two samples for one to be\
                considered as in the neighborhood of the other. This is not a\
                maximum bound on the distances of points within a cluster. This is\
                the most important DBSCAN parameter to choose appropriately for your\
                data set and distance function.",
                default=0.5,
                min=0.0001,
                # TODO check constraints
            ),
            IntegerParameter(
                name="min_samples",
                display_name="Minimum Samples",
                description="The number of samples (or total weight) in a\
                neighborhood for a point to be considered as a core point. This\
                includes the point itself.",
                default=5,
                min=1,
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
                    "cityblock": {
                        "display_name": "Cityblock",
                        "value": "cityblock",
                    },
                },
                # TODO: add more options
            ),
            OptionsParameter[str](
                name="algorithm",
                display_name="Algorithm",
                description="The algorithm to be used when computing pointwise distances\
                and finding nearest neighbors.",
                default="auto",
                is_multi=False,
                options={
                    "auto": {
                        "display_name": "Auto",
                        "value": "auto",
                    },
                    "ball_tree": {
                        "display_name": "Ball Tree",
                        "value": "ball_tree",
                    },
                    "kd_tree": {
                        "display_name": "k-d Tree",
                        "value": "kd_tree",
                    },
                    "brute": {
                        "display_name": "Brute",
                        "value": "brute",
                    },
                },
            ),
            IntegerParameter(
                name="leaf_size",
                display_name="Leaf Size",
                description="Leaf size passed to the Ball Tree or k-d Tree methods.\
                This can affect the speed of the construction and query. The optimal\
                value depends on the nature of the problem.",
                default=30,
                min=1,
                # TODO check constraints
            ),
            IntegerParameter(
                name="p",
                display_name="Exponent Value",
                description="The power of the Minkowski metric to be used to calculate distance\
                between points.",
                default=2,
                # TODO check constraints
            ),
        ]

        return Parameters(
            name=Model.dbscan,
            display_name="DBSCAN",
            parameters=parameters,
        )

    def _initialize_model(self):
        from sklearn.cluster import DBSCAN as _DBSCAN

        
        self.model = _DBSCAN(
            **self.parameters,
            n_jobs=1,
        )
