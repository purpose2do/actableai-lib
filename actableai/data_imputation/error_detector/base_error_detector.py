from abc import ABC, abstractmethod

import pandas as pd

from actableai.data_imputation.error_detector.cell_erros import CellErrors
from actableai.data_imputation.type_recon.type_detector import DfTypes


class BaseErrorDetector(ABC):
    def __init__(self):
        self._df: pd.DataFrame = NotImplemented
        self._dftypes: DfTypes = NotImplemented

    def setup(self, df: pd.DataFrame, dftypes: DfTypes):
        self._df = df
        self._dftypes = dftypes

    @abstractmethod
    def detect_cells(self) -> CellErrors:
        raise NotImplementedError
