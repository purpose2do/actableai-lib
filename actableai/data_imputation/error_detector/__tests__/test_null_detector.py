import numpy as np
import pandas as pd
import pytest

from actableai.data_imputation.error_detector.cell_erros import CellError, ErrorType
from actableai.data_imputation.error_detector.null_detector import NullDetector
from actableai.data_imputation.meta.types import ColumnType
from actableai.data_imputation.type_recon.type_detector import DfTypes


@pytest.fixture(autouse=True)
def df():
    return pd.DataFrame.from_dict(
        {
            "first_set": [
                1,
                2,
                np.NaN,
                4,
                5,
                np.nan,
                6,
                7,
            ],
            "second_set": [
                "a",
                "b",
                np.nan,
                None,
                "c",
                "d",
                "e",
                pd.NA,
            ],
            "num_with_tag": [
                "21 p",
                "50 p",
                "20 p",
                "32 p",
                "20 p",
                pd.NA,
                np.nan,
                None,
            ],
        }
    )


@pytest.fixture(autouse=True)
def df_types():
    return DfTypes(
        [
            ("first_set", ColumnType.Integer),
            ("second_set", ColumnType.String),
            ("num_with_tag", ColumnType.NumWithTag),
        ]
    )


def test_detect_error(df, df_types):
    detector = NullDetector()
    detector.setup(
        df,
        df_types,
    )
    errors = detector.detect_cells()
    assert len(errors) == 14
    errors_iter = iter(sorted(errors, key=lambda x: f"{x.column}_{x.index}"))
    assert next(errors_iter) == CellError(
        column="__num_with_tag_ltag__", index=5, error_type=ErrorType.NULL
    )
    assert next(errors_iter) == CellError(
        column="__num_with_tag_ltag__", index=6, error_type=ErrorType.NULL
    )
    assert next(errors_iter) == CellError(
        column="__num_with_tag_ltag__", index=7, error_type=ErrorType.NULL
    )
    assert next(errors_iter) == CellError(
        column="__num_with_tag_num__", index=5, error_type=ErrorType.NULL
    )
    assert next(errors_iter) == CellError(
        column="__num_with_tag_num__", index=6, error_type=ErrorType.NULL
    )
    assert next(errors_iter) == CellError(
        column="__num_with_tag_num__", index=7, error_type=ErrorType.NULL
    )
    assert next(errors_iter) == CellError(
        column="__num_with_tag_rtag__", index=5, error_type=ErrorType.NULL
    )
    assert next(errors_iter) == CellError(
        column="__num_with_tag_rtag__", index=6, error_type=ErrorType.NULL
    )
    assert next(errors_iter) == CellError(
        column="__num_with_tag_rtag__", index=7, error_type=ErrorType.NULL
    )
    assert next(errors_iter) == CellError(
        column="first_set", index=2, error_type=ErrorType.NULL
    )
    assert next(errors_iter) == CellError(
        column="first_set", index=5, error_type=ErrorType.NULL
    )
    assert next(errors_iter) == CellError(
        column="second_set", index=2, error_type=ErrorType.NULL
    )
    assert next(errors_iter) == CellError(
        column="second_set", index=3, error_type=ErrorType.NULL
    )
    assert next(errors_iter) == CellError(
        column="second_set", index=7, error_type=ErrorType.NULL
    )
