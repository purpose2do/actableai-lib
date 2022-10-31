from typing import Optional
from autogluon.core.models.abstract.abstract_model import gc
import numpy as np
from autogluon.core.models import AbstractModel
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
import dill
from autogluon.common.savers import save_pkl
from autogluon.common.loaders import load_pkl
import sys


class TabPFNModel(AbstractModel):
    """TabPFN classification Model

    Args:
        AbstractModel: Base class for all AutoGluon models.
    """

    def __init__(self, **kwargs) -> None:
        """See https://scikit-garden.github.io/api/#extratreesquantileregressor
        for more information on the parameters.
        """
        super().__init__(**kwargs)
        self.model = TabPFNClassifier()

    def _preprocess(self, X, **kwargs):
        X = super()._preprocess(X, **kwargs)
        X = X.fillna(0).to_numpy(dtype=np.float32)
        return X

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
        gc.collect()  # Try to avoid OOM error
        return sys.getsizeof(dill.dumps(self, protocol=4))
