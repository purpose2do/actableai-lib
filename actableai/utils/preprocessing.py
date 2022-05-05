import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin, BaseEstimator, _OneToOneFeatureMixin
from autogluon.features import DatetimeFeatureGenerator
from pandas import DataFrame
import pandas as pd
from pandas.api.types import is_string_dtype

def impute_df(df, numeric_imputer=None, categorical_imputer=None):
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns
    if numeric_imputer is None:
        numeric_imputer = SimpleImputer(strategy="constant", fill_value=0)
    if categorical_imputer is None:
        categorical_imputer = SimpleImputer(strategy="constant", fill_value="NA")
    if len(numeric_cols) > 0:
        df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
    if len(categorical_cols) > 0:
        df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

class TimeZoneTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X):
        return self

    def transform(self, X):
        self.result = X.apply(lambda x: x.dt.tz_convert(None))
        return self.result

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def get_feature_names(self):
        return self.result.columns


class CopyTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X):
        return self

    def transform(self, X, y=None):
        return X.copy()


class PercentageTransformer(_OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """Percentage Transformer that transforms strings with percentages into floats

    Args:
        BaseEstimator (BaseEstimator): SKLearn BaseEstimator
        TransformerMixin (TransformerMixin): SKLearn TransformerMixin
    """

    def transform(self, X, y=None):
        return X.apply(
            lambda x: x.str.extract(
                r"^[^\S\r\n]*(\d+(?:\.\d+)?)[^\S\r\n]*%[^\S\r\n]*$"
            )[0]
        ).astype(float)

    @staticmethod
    def selector(df):
        obj_mask = df.apply(is_string_dtype)
        df = df.loc[:, obj_mask]
        parsed_rate_check = lambda x, min : x.isna().sum() >= min * len(x) if x is not None else False
        extracted = df.apply(lambda x: x.str.extract(r'^[^\S\r\n]*(\d+(?:\.\d+)?)[^\S\r\n]*%[^\S\r\n]*$')[0] if hasattr(x, 'str') else None)
        val = ~extracted.apply(lambda x: parsed_rate_check(x, 0.5))
        return val[val].index

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y)


class SKLearnAGFeatureWrapperBase(TransformerMixin, BaseEstimator):
    """SKLearn Transformer Wrapper around AutoGluonFeature Generator

    Args:
        TransformerMixin (_type_): _description_
        BaseEstimator (_type_): _description_
    """

    def __init__(self, ag_feature_generator) -> None:
        """Init class

        Args:
            ag_feature_generator (AbstractFeatureGenerator): AutoGluon Feature Generator
        """
        super().__init__()
        self.ag_feature_generator = ag_feature_generator
        self.transformed_df = None

    def fit(self, X, **kwargs):
        return self.ag_feature_generator.fit(X)

    def fit_transform(self, X, y=None, **fit_params):
        self.transformed_df = self.ag_feature_generator.fit_transform(X)
        return self.transformed_df

    def transform(self, X, y=None):
        self.transformed_df = self.ag_feature_generator.transform(X)
        return self.transformed_df

    def get_feature_names_out(self, input_features=None):
        if self.transformed_df is None:
            raise Exception("Needs to be fit_transform first")
        return list(self.transformed_df.columns)

class CustomeDateTimeFeatureGenerator(DatetimeFeatureGenerator):
    def _generate_features_datetime(self, X: DataFrame) -> DataFrame:
        X_datetime = DataFrame(index=X.index)
        for datetime_feature in self.features_in:
            # TODO: Be aware: When converted to float32 by downstream models, the seconds value will be up to 3 seconds off the true time due to rounding error. If seconds matter, find a separate way to generate (Possibly subtract smallest datetime from all values).
            X_datetime[datetime_feature] = pd.to_datetime(X[datetime_feature])
            X_datetime[datetime_feature + "_year"] = X_datetime[datetime_feature].dt.year
            X_datetime[datetime_feature + "_month"] = X_datetime[datetime_feature].dt.month_name()
            X_datetime[datetime_feature + "_day"] = X_datetime[datetime_feature].dt.day
            X_datetime[datetime_feature + "_day_of_week"] = X_datetime[datetime_feature].dt.day_name()
            X_datetime[datetime_feature] = pd.to_numeric(X_datetime[datetime_feature])  # TODO: Use actual date info
            # X_datetime[datetime_feature] = pd.to_timedelta(X_datetime[datetime_feature]).dt.total_seconds()
            # TODO: Add fastai date features
        return X_datetime
