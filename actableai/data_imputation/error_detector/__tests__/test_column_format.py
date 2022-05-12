import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, call

from actableai.data_imputation.error_detector.column_format import (
    MatchRules,
    MatchStrRule,
    MatchNumRule,
    MatchRule,
)
from actableai.data_imputation.error_detector.match_condition import ConditionOp
from actableai.data_imputation.meta.types import ColumnType
from actableai.data_imputation.type_recon.type_detector import DfTypes


class TestMatchStrRule:
    @pytest.mark.parametrize(
        "series, match_str, is_regex, op, expect_indexes",
        [
            (
                pd.Series(["a", "ab", "a", "ac", "a"]),
                "a",
                False,
                ConditionOp.EQ,
                [0, 2, 4],
            ),
            (
                pd.Series(["a", "ab", "a", "ac", "a"]),
                "a",
                False,
                ConditionOp.IQ,
                [1, 3],
            ),
            (
                pd.Series(["a", np.nan, "a", "nan", "a"]),
                "nan",
                False,
                ConditionOp.EQ,
                [3],
            ),
            (
                pd.Series(["a", np.nan, "a", "", "a"]),
                "nan",
                False,
                ConditionOp.EQ,
                [],
            ),
            (pd.Series([1, 2, 1.3, 30, 2]), "1.3", False, ConditionOp.EQ, [2]),
            (
                pd.Series([1, 2, 1.3, 30, 2]),
                r"\d\.\d",
                True,
                ConditionOp.EQ,
                [2],
            ),
            (
                pd.Series([1, 2, 1.3, 30, 2]),
                r"\d{2}",
                True,
                ConditionOp.EQ,
                [3],
            ),
            (
                pd.Series([True, False, True]),
                "True",
                False,
                ConditionOp.EQ,
                [0, 2],
            ),
            (
                pd.Series(["a", "b", "c", "ab"]),
                ".",
                True,
                ConditionOp.EQ,
                [0, 1, 2],
            ),
            (
                pd.Series(["a"]),
                "a",
                False,
                "whatever",
                [],
            ),
        ],
    )
    def test_find_misplaced(self, series, match_str, is_regex, op, expect_indexes):
        rule = MatchStrRule(
            column="any column", match_str=match_str, is_regex=is_regex, op=op
        )
        indexes = rule.find_misplaced(series)
        assert list(indexes) == expect_indexes


class TestMatchNumRule:
    @pytest.mark.parametrize(
        "series, match_val, op, expect_indexes",
        [
            (pd.Series([1, 2, 1.3, 30, np.nan, 2]), 1.3, ConditionOp.EQ, [2]),
            (
                pd.Series([1, 2, 1.3, 30, np.nan, 2]),
                1.3,
                ConditionOp.IQ,
                [0, 1, 3, 5],
            ),
            (pd.Series([np.nan, 1, 2, 1.3, 30, 2]), 1.3, ConditionOp.LT, [1]),
            (
                pd.Series([1, 2, 1.3, np.nan, 30, 2]),
                1.3,
                ConditionOp.GT,
                [1, 4, 5],
            ),
            (
                pd.Series([1, 2, 1.3, 30, np.nan, 2]),
                1.3,
                ConditionOp.LTE,
                [0, 2],
            ),
            (
                pd.Series([1, np.nan, 2, 1.3, 30, 2]),
                1.3,
                ConditionOp.GTE,
                [2, 3, 4, 5],
            ),
        ],
    )
    def test_find_misplaced(self, series, match_val, op, expect_indexes):
        rule = MatchNumRule(column="any column", match_val=match_val, op=op)
        indexes = rule.find_misplaced(series)
        assert list(indexes) == expect_indexes


class TestMatchRules:
    def test_iter(self):
        rule1 = MagicMock()
        rule2 = MagicMock()
        rules = MatchRules([rule1, rule2])
        assert list(rules) == [(rule1.column, rule1), (rule2.column, rule2)]

    def test_append(self):
        rule1 = MagicMock()
        rule2 = MagicMock()
        rule3 = MagicMock()
        rule4 = MagicMock()
        rules = MatchRules([rule1, rule2])
        rules.append(rule3)
        rules.append(rule4)
        assert list(rules) == [
            (rule1.column, rule1),
            (rule2.column, rule2),
            (rule3.column, rule3),
            (rule4.column, rule4),
        ]

    @patch("actableai.data_imputation.error_detector.column_format.MatchRule.parse")
    def test_parse(self, mock_parse):
        mock_df_types = MagicMock(DfTypes)
        MatchRules.parse(mock_df_types, "1 OR 2 OR 3")
        assert mock_parse.call_args_list == [
            call(mock_df_types, "1"),
            call(mock_df_types, "2"),
            call(mock_df_types, "3"),
        ]


@pytest.fixture(autouse=True)
def df_types():
    return DfTypes(
        [
            ("integer", ColumnType.Integer),
            ("float", ColumnType.Float),
            ("string", ColumnType.String),
            ("num_with_tag", ColumnType.NumWithTag),
            ("temperature", ColumnType.Temperature),
            ("percentage", ColumnType.Percentage),
        ]
    )


class TestMatchRule:
    @pytest.mark.parametrize(
        "condition_string, expect_rule",
        [
            ("integer=100", MatchNumRule("integer", ConditionOp.EQ, float(100))),
            ("float=100.", MatchNumRule("float", ConditionOp.EQ, float(100))),
            ("float=100.1", MatchNumRule("float", ConditionOp.EQ, float(100.1))),
            (
                "num_with_tag>10",
                MatchNumRule("__num_with_tag_num__", ConditionOp.GT, float(10)),
            ),
            (
                "percentage>10.3",
                MatchNumRule("__percentage_num__", ConditionOp.GT, float(10.3)),
            ),
            (
                "temperature>10.2",
                MatchNumRule("__temperature_num__", ConditionOp.GT, float(10.2)),
            ),
            (
                "string<>any string",
                MatchStrRule("string", ConditionOp.IQ, "any string", False),
            ),
            (
                "string=s/any string/g",
                MatchStrRule("string", ConditionOp.EQ, "any string", True),
            ),
            ("string<any string", NotImplemented),
            ("integer", NotImplemented),
            ("integer=100_", NotImplemented),
            ("integer=100=2", NotImplemented),
            ("integer==100_", NotImplemented),
            ("num_with_tag>100_", NotImplemented),
        ],
    )
    def test_parse(self, df_types, condition_string, expect_rule):
        assert MatchRule.parse(df_types, condition_string) == expect_rule
