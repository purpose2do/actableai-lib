import pandas as pd
from typing import Tuple

from actableai.data_imputation.auto_fixer.fix_info import (
    FixInfoList,
)
from actableai.data_imputation.auto_fixer.refiner import Refiner
from actableai.data_imputation.config import UNABLE_TO_FIX_PLACEHOLDER, logger
from actableai.data_imputation.error_detector import ErrorDetector
from actableai.data_imputation.error_detector.cell_erros import (
    CellErrors,
)
from actableai.data_imputation.error_detector.constraint import Constraints
from actableai.data_imputation.error_detector.validation_constraints_checker import (
    ValidationConstrainsChecker,
)
from actableai.data_imputation.error_detector.validation_detector import (
    ValidationDetector,
)
from actableai.data_imputation.type_recon.type_detector import (
    ColumnType,
    DfTypes,
)


class ValidationRefiner(Refiner):
    def __init__(self, error_detector: ErrorDetector):
        super().__init__(error_detector)
        self._validation_checker: ValidationConstrainsChecker = NotImplemented
        self._custom_constraints: Constraints = NotImplemented
        for detector in error_detector.detectors:
            if isinstance(detector, ValidationDetector):
                self._custom_constraints = detector.constraints
                self._validation_checker = ValidationConstrainsChecker(
                    self._custom_constraints
                )

    def _detect_not_satisfied_cells(
        self,
        df: pd.DataFrame,
    ) -> CellErrors:
        column_types = DfTypes([(col, ColumnType.Category) for col in df.columns])
        errors = self._validation_checker.detect_typo_cells(df)

        return CellErrors(column_types, errors)

    def _find_all_value_pairs_satisfy_constraints(
        self, errors: CellErrors, df: pd.DataFrame
    ) -> pd.DataFrame:
        taboo_index = set([err.index for err in errors])
        return (
            df[~df.index.isin(taboo_index)][self._custom_constraints.mentioned_columns]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def refine(
        self,
        processed_fixed_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, FixInfoList]:
        if self._custom_constraints is NotImplemented:
            logger.info("Refine stage do not find customized restraints, skip")
            return processed_fixed_df, FixInfoList([])

        logger.info("Refining results to match custom constraints")
        not_satisfied_errors = self._detect_not_satisfied_cells(processed_fixed_df)
        logger.info(
            f"Found {len(not_satisfied_errors)} cells not match user defined constraints"
        )

        all_possible_fixes = self._find_all_possible_fixes(
            not_satisfied_errors,
            processed_fixed_df,
        )

        self._replace_best_fix_in_place(
            not_satisfied_errors, processed_fixed_df, all_possible_fixes
        )

        still_not_happy_errors = self._detect_not_satisfied_cells(processed_fixed_df)
        self._replace_as_unable_to_fix_in_place(
            still_not_happy_errors, processed_fixed_df, all_possible_fixes
        )
        logger.info(
            f"We still have {len(still_not_happy_errors)} cells not able to match user defined constraints, "
            f"they are replaced as {UNABLE_TO_FIX_PLACEHOLDER}"
        )

        return processed_fixed_df, all_possible_fixes
