import pandas as pd

from actableai.stats import Stats


class TestStats:
    def test__is_numeric(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        result = Stats()._is_numeric(df, "a")
        assert result is True

    def test__is_categorical(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        result = Stats()._is_categorical(df, "b")
        assert result is True

    def test_corr(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
        result = Stats().corr(
            df=df,
            target_col="a",
            p_value=0.05,
            categorical_columns=[],
            gen_categorical_columns=[],
        )
        assert result == [{"col": "b", "corr": 1.0, "pval": 0.0}]
