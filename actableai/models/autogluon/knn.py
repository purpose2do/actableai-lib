from actableai.models.autogluon.base import BaseParams, Model
from actableai.parameters.numeric import IntegerParameter
from actableai.parameters.options import OptionsParameter
from actableai.parameters.parameters import Parameters


class KNNParams(BaseParams):
    """Parameter class for KNN Model."""

    supported_problem_types = ["regression", "binary", "multiclass"]
    _autogluon_name = "KNN"
    explain_samples_supported = False

    @classmethod
    def _get_hyperparameters(cls, *, problem_type: str, **kwargs) -> Parameters:
        """Returns the hyperparameters space of the model.

        Args:
            problem_type: Defines the type of the problem (e.g. regression,
                binary classification, etc.). See
                cls.supported_problem_types
                for list of accepted strings

        Returns:
            The hyperparameters space.
        """
        # See
        #   Classification: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        #   Regression: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
        # TODO: Check if any additional parameters should be added
        # TODO: Disable for regression?
        # NOTE: Hyperparameter tuning is disabled for this model in AutoGluon.
        parameters = [
            IntegerParameter(
                name="n_neighbors",
                display_name="Number of Neighbors",
                description="Number of neighbors to use by default for K-neighbor queries. Must be greater than the number of samples.",
                default=5,
                min=1,
                # TODO check constraints (include max?)
            ),
            OptionsParameter[str](
                name="weights",
                display_name="Prediction Weight Function",
                description="Weight function used in prediction. For 'Uniform', all points in each neighborhood are weighted equally; for 'Distance-based', points are weighted by the inverse of their distance such that closer neighbors of a query point will have a greater influence than neighbors which are further away.",
                default="uniform",
                is_multi=False,
                options={
                    "uniform": {"display_name": "Uniform", "value": "uniform"},
                    "distance": {
                        "display_name": "Distance-based",
                        "value": "distance",
                    },
                },
            ),
            OptionsParameter[str](
                name="algorithm",
                display_name="Nearest Neighbors Algorithm",
                description="Algorithm used to compute the nearest neighbors.",
                default="auto",
                is_multi=False,
                options={
                    "auto": {"display_name": "Auto", "value": "auto"},
                    "ball_tree": {"display_name": "BallTree", "value": "ball_tree"},
                    "kd_tree": {"display_name": "KDTree", "value": "kd_tree"},
                    "brute": {
                        "display_name": "Brute-force search",
                        "value": "brute",
                    },
                },
                # NOTE: fitting on sparse input will override the setting of this parameter, using brute force.
            ),
            IntegerParameter(
                name="leaf_size",
                display_name="Leaf Size",
                description="Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.",
                default=30,
                min=1,
                max=1001,
                # TODO check constraints (include max?)
            ),
            IntegerParameter(
                name="p",
                display_name="Minkowski Metric Power",
                description="Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2.",
                default=2,
                min=1,
                # TODO check constraints (include max?)
            ),
            OptionsParameter[str](
                name="metric",
                display_name="Metric",
                description="Metric used to compute the linkage. 'Minkowski' results in the standard Euclidean distance when the 'Minkowski Metric Power' value is equal to 2.",
                default="minkowski",
                is_multi=False,
                options={
                    "minkowski": {
                        "display_name": "Minkowski",
                        "value": "minkowski",
                    },
                    "euclidean": {
                        "display_name": "Euclidean",
                        "value": "euclidean",
                    },
                    "l1": {"display_name": "L1", "value": "l1"},
                    "l2": {"display_name": "L2", "value": "l2"},
                    "manhattan": {
                        "display_name": "Manhattan",
                        "value": "manhattan",
                    },
                    "cosine": {"display_name": "Cosine", "value": "cosine"},
                },
                # TODO: should other metrics such as 'precomputed' be included?
                # See
                # https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
                # and
                # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics
                # for list of valid metrics values.
            ),
        ]

        return Parameters(
            name=Model.knn,
            display_name="K-Nearest Neighbors",
            parameters=parameters,
        )

    # TODO: Add function to ensure that n_neighbors <= n_samples for knn
    # def check_hyperparameters(self, options):
    #     s = 1
    #     # if hyperparameters['knn']['n_neighbors'] > n_samples:
    #     #         hyperparameters['knn']['n_neighbors'] = n_samples
