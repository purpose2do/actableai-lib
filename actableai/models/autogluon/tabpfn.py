from __future__ import annotations

from typing import TYPE_CHECKING, Union, Type

if TYPE_CHECKING:
    from autogluon.core.models import AbstractModel

from actableai.parameters.parameters import Parameters
from actableai.models.autogluon.base import BaseParams, Model


class TabPFNParams(BaseParams):
    """Parameter class for TabPFN Model."""

    # TODO: Check supported problem types
    supported_problem_types = ["binary", "multiclass"]
    _autogluon_name = "invalid"
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
            name=Model.tabpfn,
            display_name="TabPFN",
            parameters=parameters,
        )

    @classmethod
    def get_autogluon_name(cls) -> Union[str, Type[AbstractModel]]:
        from actableai.classification.models import TabPFNModel

        return TabPFNModel
