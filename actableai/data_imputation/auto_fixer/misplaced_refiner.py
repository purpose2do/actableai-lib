from typing import Tuple

import pandas as pd

from actableai.data_imputation.auto_fixer.fix_info import (
    FixInfoList,
)
from actableai.data_imputation.config import logger
from actableai.data_imputation.error_detector import CellErrors, ErrorDetector
from actableai.data_imputation.auto_fixer.refiner import Refiner
from actableai.data_imputation.error_detector.misplaced_detector import (
    MisplacedDetector,
)


class MisplacedRefiner(Refiner):
    def __init__(self, error_detector: ErrorDetector):
        super().__init__(error_detector)
        self._misplaced_detector: MisplacedDetector = NotImplemented
        for detector in error_detector.detectors:
            if isinstance(detector, MisplacedDetector):
                self._misplaced_detector = detector

    def _detect_not_satisfied_cells(self, df: pd.DataFrame) -> CellErrors:
        self._misplaced_detector.update_df(df)
        return self._misplaced_detector.detect_cells()

    def _find_all_value_pairs_satisfy_constraints(
        self, errors: CellErrors, df: pd.DataFrame
    ) -> pd.DataFrame:
        taboo_index = set([err.index for err in errors])
        return (
            df[~df.index.isin(taboo_index)][self._misplaced_detector.mentioned_columns]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def refine(
        self,
        processed_fixed_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, FixInfoList]:
        if self._misplaced_detector is NotImplemented:
            logger.info("Refine stage do not find misplaced detector, skip")
            return processed_fixed_df, FixInfoList([])

        not_satisfied_errors = self._detect_not_satisfied_cells(processed_fixed_df)

        all_possible_fixes = self._find_all_possible_fixes(
            not_satisfied_errors,
            processed_fixed_df,
        )

        self._replace_best_fix_in_place(
            not_satisfied_errors, processed_fixed_df, all_possible_fixes
        )

        return processed_fixed_df, all_possible_fixes
