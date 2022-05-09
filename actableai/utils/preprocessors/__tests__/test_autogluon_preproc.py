import pandas as pd
from actableai.utils.preprocessors.autogluon_preproc import (
    CustomeDateTimeFeatureGenerator,
)


class TestCustomeDateTimeFeatureGenerator:
    def test_custome_datetime_feature_generator(self):
        cdtfg = CustomeDateTimeFeatureGenerator()
        df = pd.DataFrame(
            {
                "datetime_feature": [
                    "2020-01-01T22:52:36.000Z",
                    "2019-03-05T18:32:26.000Z",
                ],
            }
        )
        df = cdtfg.fit_transform(df)
        assert len(cdtfg.features_in) == 1
        assert list(df.columns) == [
            "datetime_feature",
            "datetime_feature_year",
            "datetime_feature_month",
            "datetime_feature_day",
            "datetime_feature_day_of_week",
        ]
        assert [str(x) for x in df.dtypes] == [
            "int64",
            "int64",
            "object",
            "int64",
            "object",
        ]
