import pandas as pd
from unittest.mock import patch, MagicMock, call

from actableai.data_imputation.auto_fixer.fix_info import FixInfoList
from actableai.data_imputation.auto_fixer.strategy import FixStrategy
from actableai.data_imputation.data import DataFrame
from actableai.data_imputation.meta.types import ColumnType


@patch("actableai.data_imputation.data.data_frame.load_data")
def test_data_frame(mock_load_data):
    cols = [
        "id",
        "string",
        "enum",
        "percent",
        "temperature",
        "integer",
        "float",
        "complex",
        "number_with_tag",
        "long_string",
    ]
    mock_load_data.return_value = pd.DataFrame({col: [] for col in cols})
    df = DataFrame("any_thing")
    assert [col for col in df.columns] == cols


@patch("actableai.data_imputation.data.data_frame.load_data")
@patch("actableai.data_imputation.data.data_frame.ErrorDetector")
def test_data_frame_detect_errors_should_not_call_set_detectors_when_does_not_pass_in_detector(
    mock_class, mock_load_data
):
    mock_detector = MagicMock()
    mock_class.return_value = mock_detector

    df = DataFrame("any_thing")
    df.detect_error()

    assert mock_detector.set_detectors.call_count == 0


@patch("actableai.data_imputation.data.data_frame.load_data")
@patch("actableai.data_imputation.data.data_frame.ErrorDetector")
def test_data_frame_detect_errors_should_call_set_detectors_when_pass_in_detector(
    mock_class, mock_load_data
):
    mock_detector = MagicMock()
    mock_class.return_value = mock_detector
    mock_any_detector_1 = MagicMock()
    mock_any_detector_2 = MagicMock()

    df = DataFrame("any_thing")
    df.detect_error(mock_any_detector_1, mock_any_detector_2)

    assert mock_detector.set_detectors.call_count == 1
    mock_detector.set_detectors.assert_called_with(
        mock_any_detector_1, mock_any_detector_2
    )


@patch("actableai.data_imputation.data.data_frame.load_data")
@patch("actableai.data_imputation.data.data_frame.get_fixer")
@patch("actableai.data_imputation.data.data_frame.determine_fix_strategy")
@patch("actableai.data_imputation.data.data_frame.TypeDetector")
@patch("actableai.data_imputation.data.data_frame.Processor")
@patch("actableai.data_imputation.data.data_frame.ValidationRefiner")
@patch("actableai.data_imputation.data.data_frame.MisplacedRefiner")
def test_data_frame_auto_fix_should_call_fix(
    mock_misplaced_refiner_mod,
    mock_validation_refiner_mod,
    mock_processor,
    mock_type_detector,
    mock_determine_fix_strategy,
    mock_get_fixer,
    mock_load_data,
):
    mock_fixer = MagicMock()
    mock_get_fixer.return_value = mock_fixer
    mock_errors = MagicMock()
    mock_detect_error = MagicMock()
    mock_detect_error.return_value = mock_errors
    mock_preprocess = MagicMock()
    mock_preprocessed_df = MagicMock()
    mock_preprocess.return_value = mock_preprocessed_df
    columns_rich_meta = MagicMock()
    column_name = MagicMock()
    columns_rich_meta.name = column_name
    columns = [columns_rich_meta]
    mock_errors.columns = columns
    mock_type_detector.return_value.detect.return_value.get_meta.return_value = (
        columns_rich_meta
    )
    mock_determine_fix_strategy.return_value = FixStrategy.AUTOGLUON
    mock_validation_refiner = MagicMock()
    mock_validation_refiner_mod.return_value = mock_validation_refiner
    mock_validation_refiner.refine.return_value = MagicMock(pd.DataFrame), MagicMock(
        FixInfoList
    )
    mock_misplaced_refiner = MagicMock()
    mock_misplaced_refiner_mod.return_value = mock_misplaced_refiner
    mock_misplaced_refiner.refine.return_value = MagicMock(pd.DataFrame), MagicMock(
        FixInfoList
    )

    df = DataFrame("any_thing")
    setattr(df, "detect_error", mock_detect_error)
    setattr(df, "_preprocess", mock_preprocess)
    df.auto_fix()

    assert mock_preprocess.call_count == 1
    assert mock_fixer.fix.call_count == 1
    assert mock_fixer.fix.call_args_list == [
        call(mock_preprocessed_df, mock_errors, columns[0])
    ]
    assert mock_processor.return_value.restore.called
    assert mock_validation_refiner.refine.call_count == 1
    assert mock_misplaced_refiner.refine.call_count == 1


@patch("actableai.data_imputation.data.data_frame.load_data")
@patch("actableai.data_imputation.data.data_frame.get_fixer")
@patch("actableai.data_imputation.data.data_frame.determine_fix_strategy")
@patch("actableai.data_imputation.data.data_frame.TypeDetector")
@patch("actableai.data_imputation.data.data_frame.Processor")
@patch("actableai.data_imputation.data.data_frame.ValidationRefiner")
@patch("actableai.data_imputation.data.data_frame.MisplacedRefiner")
def test_data_frame_auto_fix_should_not_call_fix(
    mock_misplaced_refiner_mod,
    mock_validation_refiner_mod,
    mock_processor,
    mock_type_detector,
    mock_determine_fix_strategy,
    mock_get_fixer,
    mock_load_data,
):
    mock_fixer = MagicMock()
    mock_get_fixer.return_value = mock_fixer
    mock_errors = MagicMock()
    mock_detect_error = MagicMock()
    mock_detect_error.return_value = mock_errors
    mock_preprocess = MagicMock()
    mock_preprocessed_df = MagicMock()
    mock_preprocess.return_value = mock_preprocessed_df
    mock_type_detector.return_value.detect.return_value = MagicMock()
    mock_validation_refiner = MagicMock()
    mock_validation_refiner_mod.return_value = mock_validation_refiner
    mock_validation_refiner.refine.return_value = MagicMock(pd.DataFrame), MagicMock(
        FixInfoList
    )
    mock_misplaced_refiner = MagicMock()
    mock_misplaced_refiner_mod.return_value = mock_misplaced_refiner
    mock_misplaced_refiner.refine.return_value = MagicMock(pd.DataFrame), MagicMock(
        FixInfoList
    )

    mock_group_by_column = [MagicMock()]
    mock_errors.group_by_column = mock_group_by_column

    mock_determine_fix_strategy.return_value = FixStrategy.UNABLE_TO_FIX

    df = DataFrame("any_thing")
    setattr(df, "detect_error", mock_detect_error)
    setattr(df, "_preprocess", mock_preprocess)
    df.auto_fix()

    assert mock_preprocess.call_count == 1
    assert mock_fixer.fix.call_count == 0
    assert mock_processor.return_value.restore.called
    assert mock_validation_refiner.refine.call_count == 1
    assert mock_misplaced_refiner.refine.call_count == 1
