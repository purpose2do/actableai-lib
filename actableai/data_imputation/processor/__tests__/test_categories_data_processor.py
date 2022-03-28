import numpy as np
import pandas as pd

from actableai.data_imputation.meta.types import ColumnType
from actableai.data_imputation.processor import CategoriesDataProcessor
from actableai.data_imputation.type_recon.type_detector import DfTypes


class TestCategoriesDataProcessor:
    def test_should_encode_and_decode_works_when_data_is_normal(self):
        processor = CategoriesDataProcessor()

        x = pd.DataFrame(
            {
                "a": ["M", "F", "M", "O", "F", "M", np.nan, "F", None],
                "b": [1, 3, 4, 6, 5, 2, 3, 4, 2],
                "c": ["18", "19", "18", None, "21", "15", np.nan, "20", "30"],
            }
        )
        column_types = DfTypes(
            [
                ("a", ColumnType.Category),
                ("b", ColumnType.Integer),
                ("c", ColumnType.Category),
            ]
        )
        encode_result = processor.encode(x, column_types)
        assert encode_result.equals(
            pd.DataFrame(
                {
                    "a": [2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 0.0, 1.0, 0.0],
                    "b": [1, 3, 4, 6, 5, 2, 3, 4, 2],
                    "c": [2.0, 3.0, 2.0, 0.0, 5.0, 1.0, 0.0, 4.0, 6.0],
                }
            )
        )
        decode_result = processor.decode(encode_result)
        assert decode_result.equals(
            pd.DataFrame.from_dict(
                {
                    "a": [
                        "M",
                        "F",
                        "M",
                        "O",
                        "F",
                        "M",
                        "F",
                        "F",
                        "F",
                    ],
                    "b": [1, 3, 4, 6, 5, 2, 3, 4, 2],
                    "c": [
                        "18",
                        "19",
                        "18",
                        "15",
                        "21",
                        "15",
                        "15",
                        "20",
                        "30",
                    ],
                },
            )
        )

    def test_should_encode_and_decode_works_when_data_missing_category_columns(
        self,
    ):
        processor = CategoriesDataProcessor()

        x = pd.DataFrame(
            {
                "b": [1, 3, 4, 6, 5, 2, 3, 4, 2],
            }
        )
        column_types = DfTypes(
            [
                ("b", ColumnType.Integer),
            ]
        )
        encode_result = processor.encode(x, column_types)
        assert encode_result.equals(x)

        decode_result = processor.decode(encode_result)
        assert decode_result.equals(x)

    def test_should_deal_with_outbound_values_and_not_restore_as_nan(self):
        processor = CategoriesDataProcessor()

        x = pd.DataFrame(
            {
                "a": ["M", "F", "M", "O", "F", "M", np.nan, "F", None],
                "b": [
                    "18",
                    "19",
                    "18",
                    "--NaN--",
                    "21",
                    "15",
                    "--NaN--",
                    "20",
                    "30",
                ],
                "c": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
            }
        )
        column_types = DfTypes(
            [
                ("a", ColumnType.Category),
                ("b", ColumnType.Category),
                ("c", ColumnType.Category),
            ]
        )
        encode_result = processor.encode(x, column_types)
        print(encode_result)

        decode_result = processor.decode(
            pd.DataFrame(
                {
                    "a": [-1, 1, 2, 3, 1, 2, 1, 2, 1, 10],
                    "b": [-1, 2, 4, 5, 10, 2, 1, 2, 4, 10],
                    "c": [-1, 1, 2, 3, 10, 5, 6, 7, 8, 10],
                }
            )
        )

        assert decode_result.equals(
            pd.DataFrame(
                {
                    "a": [
                        "F",
                        "F",
                        "M",
                        "O",
                        "F",
                        "M",
                        "F",
                        "M",
                        "F",
                        "O",
                    ],
                    "b": [
                        "15",
                        "18",
                        "20",
                        "21",
                        "30",
                        "18",
                        "15",
                        "18",
                        "20",
                        "30",
                    ],
                    "c": ["1", "2", "3", "4", "9", "6", "7", "8", "9", "9"],
                }
            )
        )
