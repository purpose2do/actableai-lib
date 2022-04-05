import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from actableai.causal.predictors import DataFrameTransformer, SKLearnWrapper

class TestDataFrameTransformer:
    def test_fit_transform(self):
        dft = DataFrameTransformer()
        df = dft.fit_transform(
            np.array([[0, 1, 2]])
        )
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 3)

    def test_fit_transform_col_names(self):
        dft = DataFrameTransformer()
        df = dft.fit_transform(
            np.array([[0, 1, 2]]), x_w_columns=['a', 'b', 'c']
        )
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 3)
        assert list(df.columns) == ['a', 'b', 'c']

class TestSKLearnWrapper:
    def test_init(self):
        sklw = SKLearnWrapper(TabularPredictor('y'), ['a', 'b'], None, "best_quality", None)
        assert sklw.ag_args_fit is None
        assert sklw.ag_predictor is None
        assert sklw.presets == "best_quality"
        assert TabularPredictor is not None
