import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

from actableai.clustering.config import MAX_UNIQUE_FEATURES_CLUSTERING


class ClusteringDataTransformer(TransformerMixin, BaseEstimator):
    """Transform numeric columns using StandardScaler and categorical columns
    using OneHotEncoder."""

    def __init__(
        self,
        drop_low_info=True,
        drop_categorical=False,
    ) -> None:
        super().__init__()
        self.drop_low_info = drop_low_info
        self.drop_categorical = drop_categorical

    def fit_transform(self, X):
        self.transformers = []
        result = []

        self.feature_links = []
        final_feature_count = 0

        for i in range(X.shape[1]):
            self.feature_links.append([])
            # Skipping columns with only unique values
            column_value = X.iloc[:, i]
            unique_column_value = np.unique(column_value)
            if self.drop_low_info and (
                len(unique_column_value) == len(column_value)  # Every values are unique
                or len(unique_column_value) == 1  # Only one value
            ):
                # Transformer that is dropping the current column
                t = FunctionTransformer(lambda x: x)
            else:
                if is_numeric_dtype(X.iloc[:, i]):
                    t = StandardScaler()
                    result.append(t.fit_transform(X.iloc[:, i : i + 1]))

                    self.feature_links[-1].append(final_feature_count)
                    final_feature_count += 1
                elif not self.drop_categorical:
                    if len(unique_column_value) > MAX_UNIQUE_FEATURES_CLUSTERING:
                        t = FeatureHasher(
                            n_features=MAX_UNIQUE_FEATURES_CLUSTERING,
                            input_type="string",
                        )
                        result.append(t.fit_transform(X.iloc[:, i]).todense())
                    else:
                        t = OneHotEncoder(sparse=False)
                        result.append(t.fit_transform(X.iloc[:, i : i + 1]))

                    new_feature_count = result[-1].shape[1]
                    self.feature_links[-1] += list(
                        range(
                            final_feature_count, final_feature_count + new_feature_count
                        )
                    )
                    final_feature_count += new_feature_count
                else:
                    t = FunctionTransformer(lambda x: x)

            self.transformers.append(t)
        if len(result) > 0:
            return np.hstack(result)

    def transform(self, X):
        result = []
        for i in range(X.shape[1]):
            x = self.transformers[i].transform(X[:, i : i + 1])
            result.append(x)
        return np.hstack(result)

    def inverse_transform(self, X):
        c0 = 0
        result = []
        for i in range(len(self.transformers)):
            if type(self.transformers[i]) is StandardScaler:
                result.append(self.transformers[i].inverse_transform(X[:, c0 : c0 + 1]))
                c0 += 1
            else:
                t = self.transformers[i]
                result.append(
                    t.inverse_transform(X[:, c0 : c0 + len(t.categories_[0])])
                )
                c0 += len(t.categories_[0])
        return np.hstack(result)
