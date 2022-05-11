import pandas as pd
from pandas import DataFrame
from autogluon.features import DatetimeFeatureGenerator


class CustomeDateTimeFeatureGenerator(DatetimeFeatureGenerator):
    def _generate_features_datetime(self, X: DataFrame) -> DataFrame:
        X_datetime = DataFrame(index=X.index)
        for datetime_feature in self.features_in:
            # TODO: Be aware: When converted to float32 by downstream models, the seconds value will be up to 3 seconds off the true time due to rounding error. If seconds matter, find a separate way to generate (Possibly subtract smallest datetime from all values).
            X_datetime[datetime_feature] = pd.to_datetime(X[datetime_feature])
            X_datetime[datetime_feature + "_year"] = X_datetime[
                datetime_feature
            ].dt.year
            X_datetime[datetime_feature + "_month"] = X_datetime[
                datetime_feature
            ].dt.month_name()
            X_datetime[datetime_feature + "_day"] = X_datetime[datetime_feature].dt.day
            X_datetime[datetime_feature + "_day_of_week"] = X_datetime[
                datetime_feature
            ].dt.day_name()
            X_datetime[datetime_feature] = pd.to_numeric(
                X_datetime[datetime_feature]
            )  # TODO: Use actual date info
            # X_datetime[datetime_feature] = pd.to_timedelta(X_datetime[datetime_feature]).dt.total_seconds()
            # TODO: Add fastai date features
        return X_datetime
