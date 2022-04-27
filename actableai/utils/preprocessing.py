import numpy as np
from sklearn.impute import SimpleImputer

def impute_df(df, numeric_imputer=None, categorical_imputer=None):
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns
    if numeric_imputer is None:
        numeric_imputer = SimpleImputer(strategy='constant', fill_value=0)
    if categorical_imputer is None:
        categorical_imputer = SimpleImputer(strategy='constant', fill_value="NA")
    if len(numeric_cols) > 0:
        df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
    if len(categorical_cols) > 0:
        df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

from sklearn.base import BaseEstimator, TransformerMixin, _OneToOneFeatureMixin
from pandas.api.types import is_string_dtype, is_bool_dtype
class PercentageTransformer(_OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """Percentage Transformer that transforms strings with percentages into floats

    Args:
        BaseEstimator (BaseEstimator): SKLearn BaseEstimator
        TransformerMixin (TransformerMixin): SKLearn TransformerMixin
    """
    def fit_transform(self, X, y=None):
        return self.transform(X, y)

    def fit(self, X):
        return self

    def transform(self, X, y=None):
        return X.apply(lambda x: x.str.extract(r'^[^\S\r\n]*(\d+(?:\.\d+)?)[^\S\r\n]*%[^\S\r\n]*$')[0]).astype(float)

    @staticmethod
    def selector(df):
        obj_mask = df.apply(is_string_dtype)
        df = df.loc[:, obj_mask]
        parsed_rate_check = lambda x, min : x.isna().sum() >= min * len(x) if x is not None else False
        extracted = df.apply(lambda x: x.str.extract(r'^[^\S\r\n]*(\d+(?:\.\d+)?)[^\S\r\n]*%[^\S\r\n]*$')[0] if hasattr(x, 'str') else None)
        val = ~extracted.apply(lambda x: parsed_rate_check(x, 0.5))
        return val[val].index
