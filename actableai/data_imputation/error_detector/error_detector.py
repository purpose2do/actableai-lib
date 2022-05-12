import pandas as pd

from actableai.data_imputation.config import logger
from actableai.data_imputation.error_detector.base_error_detector import (
    BaseErrorDetector,
)
from actableai.data_imputation.error_detector.cell_erros import (
    CellErrors,
)
from actableai.data_imputation.error_detector.misplaced_detector import (
    MisplacedDetector,
)
from actableai.data_imputation.error_detector.null_detector import NullDetector
from actableai.data_imputation.error_detector.validation_detector import (
    ValidationDetector,
)
from actableai.data_imputation.type_recon.type_detector import DfTypes


class ErrorDetector:
    def __init__(self):
        self._detectors = [NullDetector(), MisplacedDetector()]

    @property
    def detectors(self):
        return self._detectors

    def set_detectors(self, *detectors: BaseErrorDetector):
        self._detectors = detectors

    def detect_error(
        self,
        df: pd.DataFrame,
        dftypes: DfTypes,
        expanded_df: pd.DataFrame,
        expanded_dftypes: DfTypes,
    ) -> CellErrors:
        errors = CellErrors(dftypes)

        for detector in self._detectors:
            if isinstance(detector, NullDetector):
                detector.setup(df, dftypes)
            elif isinstance(detector, ValidationDetector):
                detector.setup(df, dftypes)
            else:
                detector.setup(expanded_df, expanded_dftypes)
            new_errors = detector.detect_cells()
            errors.extend(new_errors)

        logger.info(f"Found {len(errors)} in total")
        return errors
