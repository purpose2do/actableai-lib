import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

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

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

class CustomColumnTransformer(ColumnTransformer):
    def get_feature_names_out(self, input_features=None):
        result = super().get_feature_names_out(input_features)
        print(result)
        for name, _, _ in self.transformers:
            result = [x.replace(name + "__", "") for x in result]
        result = [x.replace("remainder__", "") for x in result]
        return result

class PercentageTransformer(BaseEstimator, TransformerMixin):
    def fit_transform(self, X, y=None):
        return self.transform(X, y)

    def fit(self, X):
        return self

    def transform(self, X, y=None):
        return X.apply(lambda x: x.str.extract(r'^[^\S\r\n]*(\d+(?:\.\d+)?)[^\S\r\n]*%[^\S\r\n]*$')[0]).astype(float)

    def get_feature_names_out(self, input_features):
        return input_features

    @staticmethod
    def predicate(df):
        obj_cols = list(df.select_dtypes(include='object').columns)
        parsed_rate_check = lambda x, min : x.isna().sum() >= min * len(x)
        extracted = df[obj_cols].apply(lambda x: x.str.extract(r'^[^\S\r\n]*(\d+(?:\.\d+)?)[^\S\r\n]*%[^\S\r\n]*$')[0])
        return extracted.apply(lambda x: parsed_rate_check(x, 0.5))
