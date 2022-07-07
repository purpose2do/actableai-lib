import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.cluster import KMeans
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from pandas.api.types import is_numeric_dtype
from actableai.clustering.config import MAX_UNIQUE_FEATURES_CLUSTERING


class ClusteringDataTransformer(TransformerMixin, BaseEstimator):
    """Transform numeric columns using StandardScaler and categorical columns
    using OneHotEncoder."""

    def fit_transform(self, X):
        self.transformers = []
        result = []

        self.feature_links = []
        final_feature_count = 0

        for i in range(X.shape[1]):
            self.feature_links.append([])
            # Skipping columns with only unique values
            if len(np.unique(X[:, i])) == len(X[:, i]):
                t = FunctionTransformer(lambda x: x)
                self.feature_links[-1].append(final_feature_count)
                final_feature_count += 1
            else:
                if is_numeric_dtype(X[:, i]):
                    t = StandardScaler()
                    result.append(t.fit_transform(X[:, i : i + 1]))

                    self.feature_links[-1].append(final_feature_count)
                    final_feature_count += 1
                else:
                    if len(np.unique(X[:, i])) > MAX_UNIQUE_FEATURES_CLUSTERING:
                        t = FeatureHasher(
                            n_features=MAX_UNIQUE_FEATURES_CLUSTERING,
                            input_type="string",
                        )
                    else:
                        t = OneHotEncoder(sparse=False)
                    result.append(t.fit_transform(X[:, i : i + 1]))

                    new_feature_count = result[-1].shape[0]
                    self.feature_links[-1] += list(
                        range(
                            final_feature_count, final_feature_count + new_feature_count
                        )
                    )
                    final_feature_count += new_feature_count

            self.transformers.append(t)
        return np.hstack(result)

    def transform(self, X):
        result = []
        for i in range(X.shape[1]):
            x = self.transformers[i].transform(X[:, i : i + 1])
            if i in self.categorical_cols:
                x = x.todense()
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


def KMeans_scaled_inertia(
    scaled_data: np.ndarray, k: int, alpha_k: float, *KMeans_args, **KMeans_kwargs
):
    """KMeans with scaled inertia.

    Args:
        scaled_data: matrix scaled data. rows are samples and columns are features for
            clustering.
        k: current k for applying KMeans.
        alpha_k: manually tuned factor that gives penalty to the number of clusters.

    Returns:
        float: scaled inertia value for current k
    """

    # fit k-means
    inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
    kmeans = KMeans(n_clusters=k, *KMeans_args, **KMeans_kwargs).fit(scaled_data)
    scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
    return scaled_inertia


def KMeans_pick_k(
    scaled_data, alpha_k, k_range, *KMeans_args, **KMeans_kwargs
) -> KMeans:
    """Find best k for KMeans based on scaled inertia method.
        https://towardsdatascience.com/an-approach-for-choosing-number-of-clusters-for-k-means-c28e614ecb2c

    Args:
        scaled_data: matrix scaled data. rows are samples and columns are features for
            clustering.
        alpha_k: manually tuned factor that gives penalty to the number of clusters.
        k_range: range of k values to test.

    Returns:
        KMeans: KMeans object with best k.
    """
    ans = []
    for k in k_range:
        scaled_inertia = KMeans_scaled_inertia(
            scaled_data, k, alpha_k, *KMeans_args, **KMeans_kwargs
        )
        ans.append((k, scaled_inertia))
    results = pd.DataFrame(ans, columns=["k", "Scaled Inertia"]).set_index("k")
    best_k = results.idxmin()[0]
    return best_k


def KMeans_pick_k_sil(X, k_range, *KMeans_args, **KMeans_kwargs):
    """Find best k for KMeans based on silhouette score.
        https://newbedev.com/scikit-learn-k-means-elbow-criterion

    Args:
        X: matrix of data. rows are samples and columns are features for
            clustering.
        k_range: range of k values to test.

    Returns:
        KMeans: KMeans object with best k.
    """
    max_sil_coeff, best_k = 0, 2
    for k in k_range:
        kmeans = KMeans(n_clusters=k).fit(X)
        label = kmeans.labels_
        sil_coeff = silhouette_score(X, label, metric="euclidean")
        print("Cluster: ", k, ", Silhouette coeff: ", sil_coeff)
        if max_sil_coeff < sil_coeff:
            max_sil_coeff = sil_coeff
            best_k = k
    return best_k
