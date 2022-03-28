import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from actableai.utils import (
    memory_efficient_hyperparameters,
    fast_categorical_hyperparameters,
)


class UnsupportedPredictorType(ValueError):
    pass


class UnsupportedProblemType(ValueError):
    pass


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, X):
        self._fitted = False
        if X is not None and type(X) is pd.DataFrame:
            self._fitted = True
            self.columns = list(X.columns)
            self.dtypes = X.dtypes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not self._fitted:
            return pd.DataFrame(X)

        X = pd.DataFrame(X, columns=self.columns)
        for col, dtype in zip(X, self.dtypes):
            X[col] = X[col].astype(dtype)
        return X


class LinearRegressionWrapper(LinearRegression):

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return {
            "r2": r2_score(y, y_pred, sample_weight=sample_weight),
            "y": y,
            "y_pred": y_pred,
            "sample_weight": sample_weight
        }


class SKLearnWrapper:
    def __init__(
        self,
        df: pd.DataFrame,
        ag_predictor: TabularPredictor,
        hyperparameters=None,
        presets="best_quality",
        ag_args_fit=None,
    ):
        """Construct a sklearn wrapper object

        Args:
            ag_predictor (TabularPredictor): Autogluon tabular predictor
            hyperparameters (dict, Optional): dictionary of hyperparameters
        """
        if type(ag_predictor) is not TabularPredictor:
            raise UnsupportedPredictorType()
        self.ag_predictor = ag_predictor
        if hyperparameters is None:
            self.hyperparameters = "default"
        else:
            self.hyperparameters = hyperparameters
        self.presets = presets
        self._df_transformer = DataFrameTransformer(df)
        self.ag_args_fit = ag_args_fit

    def fit(self, X, y, sample_weight=None):
        label = self.ag_predictor.label
        train_data = self._df_transformer.transform(X)
        train_data[label] = y
        train_data = TabularDataset(train_data)
        self.ag_predictor.sample_weight = sample_weight

        self.ag_predictor.fit(
            train_data=train_data,
            presets=self.presets,
            hyperparameters=self.hyperparameters,
            ag_args_fit=self.ag_args_fit or {},
        )
        self.train_data = train_data

    def feature_importance(self):
        return self.ag_predictor.feature_importance(self.train_data)

    def predict(self, X):
        test_data = self._df_transformer.transform(X)
        test_data = TabularDataset(test_data)
        y_pred = self.ag_predictor.predict(test_data).values
        return y_pred

    def predict_proba(self, X):
        test_data = self._df_transformer.transform(X)
        y_pred_proba = self.ag_predictor.predict_proba(test_data)
        return y_pred_proba.values

    def score(self, X, y, sample_weight=None):
        test_data = self._df_transformer.transform(X)
        y_pred = self.ag_predictor.predict(test_data).values
        if self.ag_predictor.problem_type in ["binary", "multiclass"]:
            return {
                "accuracy": accuracy_score(y, y_pred, sample_weight=sample_weight),
                "y": y,
                "y_pred": y_pred,
                "sample_weight": sample_weight
            }
        elif self.ag_predictor.problem_type == "regression":
            return {
                "r2": r2_score(y, y_pred, sample_weight=sample_weight),
                "y": y,
                "y_pred": y_pred,
                "sample_weight": sample_weight
            }
        else:
            raise UnsupportedProblemType()
