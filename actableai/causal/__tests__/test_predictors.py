import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
import pytest

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
        assert sklw.ag_predictor is not None
        assert sklw.presets == "best_quality"
        assert TabularPredictor is not None

    def test_fit(self):
        sklw = SKLearnWrapper(TabularPredictor('y'), ['a', 'b', 'c'], None, "best_quality", None)
        X = np.array([
            [0, 1, 2],
            [0, 1, 2]
        ])
        y = np.array([0, 1])
        sklw.fit(X, y)
        sklw.ag_predictor.sample_weight is not None
        sklw.train_data is not None

    def test_feature_importance(self):
        sklw = SKLearnWrapper(TabularPredictor('y'), ['a', 'b', 'c'], None, "best_quality", None)
        X = np.array([
            [0, 1, 2],
            [0, 1, 2]
        ])
        y = np.array([0, 1])
        sklw.fit(X, y)
        feat_imp = sklw.feature_importance()
        feat_imp is not None

    def test_feature_importance_no_fit(self):
        sklw = SKLearnWrapper(TabularPredictor('y'), ['a', 'b', 'c'], None, "best_quality", None)
        with pytest.raises(Exception):
            sklw.feature_importance()
