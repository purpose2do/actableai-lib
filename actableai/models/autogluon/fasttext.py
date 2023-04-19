from actableai.models.autogluon.base import BaseParams, Model
from actableai.parameters.parameters import Parameters


class FASTTEXTParams(BaseParams):
    """Parameter class for FASTTEXT Model."""

    # See https://auto.gluon.ai/0.5.2/_modules/autogluon/tabular/models/fasttext/fasttext_model.html

    # TODO: Check supported problem types
    supported_problem_types = ["binary", "multiclass"]
    _autogluon_name = "FASTTEXT"
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
        parameters = []

        return Parameters(
            name=Model.fasttext,
            display_name="FastText",
            parameters=parameters,
        )
