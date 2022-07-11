import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher

from actableai.clustering import ClusteringDataTransformer


class TestClusteringDataTransformer:
    def test_clustering_data_transformer(self):
        X = pd.DataFrame(
            [[1.0, "a", "@", 7.0], [2.0, "b", "#", 8.0], [3.0, "c", "$", 9.0]]
        )
        t = ClusteringDataTransformer(drop_low_info=False)
        transformed_x = t.fit_transform(X)
        assert np.all(
            np.round(transformed_x, decimals=2)
            == np.array(
                [
                    [-1.22, 1, 0, 0, 0, 0, 1, -1.22],
                    [0, 0, 1, 0, 1, 0, 0, 0],
                    [1.22, 0, 0, 1, 0, 1, 0, 1.22],
                ]
            )
        )
        inversed_x = t.inverse_transform(transformed_x)
        assert np.all(inversed_x == X)

    def test_clustering_data_transformer_drop_low_info(self):
        X = pd.DataFrame(
            [[1.0, "a", "@", 7.0], [2.0, "b", "#", 8.0], [3.0, "c", "$", 9.0]]
        )
        t = ClusteringDataTransformer(drop_low_info=True)
        transformed_x = t.fit_transform(X)
        assert transformed_x is None

    def test_clustering_data_transformer_high_cardinality(self):
        X = pd.DataFrame([[str(i)] for i in range(1500)])
        t = ClusteringDataTransformer(drop_low_info=False)
        transformed_x = t.fit_transform(X)
        assert isinstance(t.transformers[0], FeatureHasher)
        assert transformed_x is not None
