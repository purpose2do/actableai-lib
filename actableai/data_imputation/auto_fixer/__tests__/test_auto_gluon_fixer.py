import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, ANY

from actableai.data_imputation.auto_fixer.auto_gluon_fixer import AutoGluonFixer
from actableai.data_imputation.error_detector import CellErrors
from actableai.data_imputation.error_detector.cell_erros import (
    CellError,
    ErrorType,
)
from actableai.data_imputation.meta.column import RichColumnMeta
from actableai.data_imputation.meta.types import ColumnType
from actableai.data_imputation.type_recon.type_detector import DfTypes
from actableai.utils import memory_efficient_hyperparameters

stub_path = "actableai.data_imputation.auto_fixer.auto_gluon_fixer"


@patch(f"{stub_path}.TabularDataset")
@patch(f"{stub_path}.TabularPredictor")
@patch(f"{stub_path}.get_df_without_error")
@patch(f"{stub_path}.time.time")
@pytest.mark.parametrize(
    "df, columns_to_train, all_errors, column_to_predict",
    [
        (
            pd.DataFrame(
                data={
                    "a": [
                        "a",
                        "b",
                        "b",
                        "b",
                        "b",
                        "a",
                        "a",
                        "a",
                        "a",
                        "a",
                        "a",
                        "a",
                        "a",
                        "a",
                        "a",
                        "a",
                    ],
                    "b": [1, 2, 2, 2, 2, None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    "c": [None, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                }
            ),
            ["a", "c"],
            CellErrors(
                DfTypes([("b", ColumnType.Integer)]),
                [
                    CellError("b", 5, ErrorType.NULL),
                    CellError("c", 0, ErrorType.NULL),
                ],
            ),
            RichColumnMeta("b", ColumnType.Integer),
        )
    ],
)
def test__predict_missing_for_single_column(
    mock_time_func,
    mock_get_df_without_error,
    mock_predictor_mod,
    mock_dataset_mod,
    df,
    columns_to_train,
    all_errors,
    column_to_predict,
):
    df = MagicMock()
    df_without_error = MagicMock()
    df_to_train = MagicMock()
    df_to_test = MagicMock()
    mock_predictor = MagicMock()
    mock_predictor_mod.return_value = mock_predictor
    mock_time = 0
    mock_time_func.return_value = mock_time

    mock_get_df_without_error.return_value = df_without_error
    df_without_error.empty = False

    mock_dataset_mod.side_effect = [df_to_train, df_to_test]

    fixer = AutoGluonFixer()
    mock_decide_problem_type_func = MagicMock()
    mock_decide_problem_type = MagicMock()
    mock_decide_problem_type_func.return_value = mock_decide_problem_type
    fixer._decide_problem_type = mock_decide_problem_type_func
    fixer._predict_missing_for_single_column(
        df,
        columns_to_train=columns_to_train,
        all_errors=all_errors,
        column_to_predict=column_to_predict,
    )

    mock_predictor_mod.assert_called_with(
        label=column_to_predict.name,
        problem_type=mock_decide_problem_type.value,
        path=f"./AutogluonModels_{mock_time}",
    )
    mock_predictor.fit.assert_called_with(
        df_to_train,
        hyperparameters=memory_efficient_hyperparameters(),
        excluded_model_types=["CAT"],
        holdout_frac=ANY,
    )
    mock_predictor.predict_proba.assert_called_with(df_to_test)
