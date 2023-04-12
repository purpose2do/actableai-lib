from functools import lru_cache

from umap import UMAP as _UMAP

from actableai.embedding.models.base import EmbeddingModelWrapper, Model
from actableai.parameters.numeric import IntegerParameter, FloatParameter
from actableai.parameters.options import OptionsParameter
from actableai.parameters.parameters import Parameters


class UMAP(EmbeddingModelWrapper):
    """Class to handle UMAP."""

    @staticmethod
    @lru_cache(maxsize=None)
    def get_parameters() -> Parameters:
        """Returns the parameters of the model.

        Returns:
            The parameters.
        """
        parameters = [
            IntegerParameter(
                name="n_neighbors",
                display_name="Neighbors Count",
                description="The size of local neighborhood (in terms of number of\
                neighboring sample points) used for manifold approximation. Larger\
                values result in more global views of the manifold, while smaller\
                values result in more local data being preserved. In general values\
                should be in the range 2 to 100.",
                default=15,
                min=1,
                # TODO check constraints
            ),
            FloatParameter(
                name="learning_rate",
                display_name="Learning Rate",
                description="Learning rate used during training.",
                default=1.0,
                min=0.0001,
                # TODO check constraints
            ),
            OptionsParameter[str](
                name="metric",
                display_name="Metric",
                description="The metric to use when calculating distance between instances in a feature array.",
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
                    # TODO add more losses, see https://umap-learn.readthedocs.io/en/latest/api.html
                },
            ),
            OptionsParameter[str](
                name="init",
                display_name="Initializing Method",
                description="How to initialize the low dimensional embedding.",
                default="spectral",
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
            FloatParameter(
                name="min_dist",
                display_name="Minimum Distance",
                description="The effective minimum distance between embedded points.\
                Smaller values will result in a more clustered/clumped embedding\
                where nearby points on the manifold are drawn closer together, while\
                larger values will result on a more even dispersal of points. The\
                value should be set relative to the spread value, which determines\
                the scale at which embedded points will be spread out.",
                default=0.1,
                # TODO check constraints
            ),
            FloatParameter(
                name="spread",
                display_name="Spread",
                description="The effective scale of embedded points. In combination with\
                min_dist this determines how clustered/clumped the embedded points\
                are.",
                default=1.0,
                # TODO check constraints
            ),
            IntegerParameter(
                name="local_connectivity",
                display_name="Local Connectivity",
                description="The local connectivity required - i.e. the\
                number of nearest neighbors that should be assumed to be connected\
                at a local level. The higher this value the more connected the\
                manifold becomes locally. In practice this should be not more than\
                the local intrinsic dimension of the manifold.",
                default=1,
                # TODO check constraints
            ),
            FloatParameter(
                name="repulsion_strength",
                display_name="Repulsion Strength",
                description=" Weighting applied to negative samples in low\
                dimensional embedding optimization. Values higher than one will\
                result in greater weight being given to negative samples.",
                default=1,
                # TODO check constraints
            ),
            IntegerParameter(
                name="negative_sample_rate",
                display_name="Negative Sample Rate",
                description="The number of negative samples to select per\
                positive sample in the optimization process. Increasing this value\
                will result in greater repulsive force being applied, greater\
                optimization cost, but slightly more accuracy.",
                default=5,
                # TODO check constraints
            ),
            FloatParameter(
                name="transform_queue_size",
                display_name="Transform Queue Size",
                description="For transform operations (embedding new points\
                using a trained model this will control how aggressively to search\
                for nearest neighbors). Larger values will result in slower\
                performance but more accurate nearest neighbor evaluation.",
                default=4,
                # TODO check constraints
            ),
            # target_metric: str = "categorical",
            OptionsParameter[str](
                name="target_metric",
                display_name="Target Metric",
                description="The metric used to measure distance for a target\
                array is using supervised dimension reduction. By default this\
                is 'categorical' which will measure distance in terms of whether\
                categories match or are different. Furthermore, if\
                semi-supervised is required target values of -1 will be treated\
                as unlabelled under the 'categorical' metric. If the target\
                array takes continuous values (e.g. for a regression problem)\
                then metric of 'l1' or 'l2' is probably more appropriate.",
                default="categorical",
                is_multi=False,
                options={
                    "categorical": {
                        "display_name": "Categorical",
                        "value": "categorical",
                    },
                    "l1": {
                        "display_name": "L1",
                        "value": "l1",
                    },
                    "l2": {
                        "display_name": "L2",
                        "value": "l2",
                    },
                },
            ),
            FloatParameter(
                name="target_weight",
                display_name="Target Weight",
                description="Weighting factor between data topology and target\
                topology. A value of 0.0 weights predominantly on data, a value of\
                1.0 places a strong emphasis on target. The default of 0.5 balances\
                the weighting equally between data and target.",
                default=0.5,
                min=0,
                max=1.0000001,
                # TODO check constraints
            ),
            IntegerParameter(
                name="random_state",
                display_name="Random State",
                description="Determines the random number generator.",
                default=0,
            ),
        ]

        return Parameters(
            name=Model.umap,
            display_name="UMAP",
            parameters=parameters,
        )

    def _initialize_model(self):
        self.model = _UMAP(
            n_components=self.embedding_size,
            **self.parameters,
            verbose=self.verbosity > 0,
            low_memory=True,
            transform_seed=self.parameters["random_state"],
            n_jobs=1,
        )
