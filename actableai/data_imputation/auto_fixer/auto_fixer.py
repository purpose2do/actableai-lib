from abc import ABC, abstractmethod

import pandas as pd

from actableai.data_imputation.auto_fixer.fix_info import FixInfoList
from actableai.data_imputation.error_detector import CellErrors
from actableai.data_imputation.meta.column import RichColumnMeta


class AutoFixer(ABC):
    @abstractmethod
    def fix(
        self,
        df: pd.DataFrame,
        all_errors: CellErrors,
        current_column: RichColumnMeta,
    ) -> FixInfoList:
        raise NotImplementedError
