from abc import ABC, abstractmethod

import pandas as pd

from actableai.data_imputation.auto_fixer.fix_info import FixInfoList
from actableai.data_imputation.error_detector import CellErrors
from actableai.data_imputation.meta.column import RichColumnMeta


class AutoFixer(ABC):
    """Abscract class for auto fixers.

    Args:
        ABC: Abstract Base Class

    Raises:
        NotImplementedError: If the method is not implemented.
    """
    @abstractmethod
    def fix(
        self,
        df: pd.DataFrame,
        all_errors: CellErrors,
        current_column: RichColumnMeta,
    ) -> FixInfoList:
        """Abstract method for fixing errors.

        Args:
            df: DataFrame to fix.
            all_errors: All errors in the dataframe.
            current_column: Current column to fix.

        Raises:
            NotImplementedError: If the method is not implemented.

        Returns:
            FixInfoList: List of fix information.
        """
        raise NotImplementedError
