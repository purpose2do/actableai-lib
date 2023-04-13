from functools import lru_cache

from actableai.clustering.models.base import ClusteringModelWrapper, Model
from actableai.parameters.numeric import (
    FloatParameter,
    IntegerParameter,
)
from actableai.parameters.parameters import Parameters


class AffinityPropagation(ClusteringModelWrapper):
    """Class to handle Affinity Propagation.

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
                name="damping",
                display_name="Damping",
                description="The extent to which the current value is maintained\
                relative to incoming values (weighted 1 - damping)",
                default=0.5,
                min=0.5,
                max=1.0,
                # TODO check constraints
            ),
            IntegerParameter(
                name="max_iter",
                display_name="Maximum Iterations",
                description="Maximum number of iterations.",
                default=200,
                min=1,
                # TODO check constraints
            ),
            IntegerParameter(
                name="convergence_iter",
                display_name="Convergence Iterations",
                description="Number of iterations with no change in the\
                number of estimated clusters that stops the convergence.",
                default=15,
                min=1,
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
            name=Model.affinity_propagation,
            display_name="Affinity Propagation",
            parameters=parameters,
        )

    def _initialize_model(self):
        from sklearn.cluster import AffinityPropagation as _AffinityPropagation

        
        self.model = _AffinityPropagation(
            **self.parameters,
            copy=True,
            verbose=self.verbosity > 0,
        )
