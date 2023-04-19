from actableai.models.autogluon.base import BaseParams, Model
from actableai.parameters.numeric import IntegerParameter
from actableai.parameters.options import OptionsSpace
from actableai.parameters.parameters import Parameters


class AGAUTOMMParams(BaseParams):
    """Parameter class for AG_AUTOMM Model."""

    # TODO: Check supported problem types
    supported_problem_types = ["regression", "binary", "multiclass"]
    _autogluon_name = "AG_AUTOMM"
    explain_samples_supported = False
    gpu_required = True

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
        # Change the batch size if we encounter memory issues

        # Parameters obtained from:
        # from autogluon.text.text_prediction.presets import list_text_presets
        # simple_presets = list_text_presets(verbose=True)
        # parameters = simple_presets["multilingual"]

        # hyperparameters["AG_AUTOMM"]["env.per_gpu_batch_size"] = 4

        parameters = [
            OptionsSpace[str](
                name="model.hf_text.checkpoint_name",
                display_name="Checkpoint Name",
                description="Checkpoint name.",
                default=["microsoft/mdeberta-v3-base"],
                hidden=True,
                options={
                    "microsoft/mdeberta-v3-base": {
                        "display_name": "microsoft/mdeberta-v3-base",
                        "value": "microsoft/mdeberta-v3-base",
                    },
                },
            ),
            IntegerParameter(
                name="optimization.top_k",
                display_name="Top-K",
                description="Top-K value.",
                default=1,
                min=0,
                hidden=True,
                # TODO check constraints
            ),
            OptionsSpace[str](
                name="env.precision",
                display_name="Precision",
                description="Precision.",
                default=["bf16"],
                hidden=True,
                options={
                    "bf16": {
                        "display_name": "bf16",
                        "value": "bf16",
                    },
                },
            ),
            IntegerParameter(
                name="env.per_gpu_batch_size",
                display_name="Per-GPU Batch Size",
                description="Per-GPU batch size.",
                default=4,
                min=0,
                hidden=True,
                # TODO check constraints
            ),
        ]

        return Parameters(
            name=Model.ag_automm,
            display_name="AG AUTOMM",
            parameters=parameters,
        )
