from unittest.mock import MagicMock

from actableai.data_imputation import (
    NullDetector,
    ValidationDetector,
    MisplacedDetector,
)
from actableai.data_imputation.error_detector import ErrorDetector


def test_error_detector_set_detectors():
    detector_1 = MagicMock(NullDetector)
    detector_2 = MagicMock(ValidationDetector)
    detector_3 = MagicMock(MisplacedDetector)
    df = MagicMock()
    expand_df = MagicMock()
    dtype = MagicMock()
    expand_dftype = MagicMock()

    d = ErrorDetector()
    d.set_detectors(detector_1, detector_3, detector_2)
    d.detect_error(df, dtype, expand_df, expand_dftype)

    assert detector_1.detect_cells.called
    detector_1.setup.assert_called_with(df, dtype)
    assert detector_2.detect_cells.called
    detector_2.setup.assert_called_with(df, dtype)
    assert detector_3.detect_cells.called
    detector_3.setup.assert_called_with(expand_df, expand_dftype)
