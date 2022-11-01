import sys
import dill
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
from autogluon.core.models import AbstractModel
from autogluon.common.savers import save_pkl
from autogluon.common.loaders import load_pkl


class TabPFNModel(AbstractModel):
    """TabPFN classification Model"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = TabPFNClassifier()

    def save(self, path: str = None, verbose=True) -> str:
        if path is None:
            path = self.path
        file_path = path + self.model_file_name
        save_pkl.save_with_fn(
            path=file_path,
            object=self,
            pickle_fn=lambda o, buffer: dill.dump(o, buffer, protocol=4),
            verbose=verbose,
        )
        return path

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        file_path = path + cls.model_file_name
        model = load_pkl.load_with_fn(
            path=file_path, pickle_fn=dill.load, verbose=verbose
        )
        if reset_paths:
            model.set_contexts(path)
        return model

    def get_memory_size(self) -> int:
        from autogluon.core.models.abstract.abstract_model import gc

        gc.collect()  # Try to avoid OOM error
        return sys.getsizeof(dill.dumps(self, protocol=4))
