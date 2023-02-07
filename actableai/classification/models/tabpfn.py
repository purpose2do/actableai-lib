import sys
import dill as pickle
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
from autogluon.core.models import AbstractModel
from autogluon.common.savers import save_pkl
from autogluon.common.loaders import load_pkl


class TabPFNModel(AbstractModel):
    """TabPFN classification Model"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.initialize()
        # Set to false so AutoGluon can re-initialize in the fit function
        self._is_initialized = False

        base_path = self.params_aux.get("tabpfn_model_directory")

        tabpfn_params = {
            "N_ensemble_configurations": 20,
        }
        if base_path is not None:
            tabpfn_params["base_path"] = base_path

        self.model = TabPFNClassifier(**tabpfn_params)

    def _get_default_auxiliary_params(self):
        default_auxiliary_params = super()._get_default_auxiliary_params()

        extra_auxiliary_params = {
            "tabpfn_model_directory": None,
        }

        default_auxiliary_params.update(extra_auxiliary_params)

        return default_auxiliary_params

    def save(self, path: str = None, verbose=True) -> str:
        if path is None:
            path = self.path
        file_path = path + self.model_file_name
        save_pkl.save_with_fn(
            path=file_path,
            object=self,
            pickle_fn=lambda o, buffer: pickle.dump(o, buffer, protocol=4),
            verbose=verbose,
        )
        return path

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        file_path = path + cls.model_file_name
        model = load_pkl.load_with_fn(
            path=file_path, pickle_fn=pickle.load, verbose=verbose
        )
        if reset_paths:
            model.set_contexts(path)
        return model

    def get_memory_size(self) -> int:
        from autogluon.core.models.abstract.abstract_model import gc

        gc.collect()  # Try to avoid OOM error
        return sys.getsizeof(pickle.dumps(self, protocol=4))
