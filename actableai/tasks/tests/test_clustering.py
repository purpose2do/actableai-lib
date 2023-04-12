import pytest

import pandas as pd

from actableai.data_validation.base import CheckLevels
from actableai.tasks.clustering import AAIClusteringTask
from actableai.utils.dataset_generator import DatasetGenerator
from actableai.clustering.models.base import Model as ClusteringModel
from actableai.embedding.models.base import Model as ProjectionModel


@pytest.fixture(scope="function")
def clustering_task():
    yield AAIClusteringTask(use_ray=False)


class TestRemoteSegmentation:
    def test_segment_1_col(self, clustering_task):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
            }
        )

        r = clustering_task.run(
            df,
            num_clusters=3,
            parameters={
                "clustering_model": {"agglomerative_clustering": {}},
                "projection_model": {"umap": {}},
            },
        )

        assert r["status"] == "SUCCESS"
        assert "data" in r
        assert "explanation" in r["data"][0]

    def test_segment_bool_col(self, clustering_task):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [True, True, False, True, False, False, True, True, False, False]
                * 2,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        r = clustering_task.run(
            df,
            num_clusters=3,
            parameters={
                "clustering_model": {"agglomerative_clustering": {}},
                "projection_model": {"linear_discriminant_analysis": {}},
            },
        )

        assert r["status"] == "SUCCESS"
        assert "data" in r
        assert "explanation" in r["data"][0]

    """
    def test_auto_cluster(self, clustering_task):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
            }
        )

        r = clustering_task.run(df, num_clusters="auto")

        assert r["status"] == "SUCCESS"
        assert "data" in r
    """

    def test_segment_mutiple_cols(self, clustering_task):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        r = clustering_task.run(
            df,
            features=["x", "y"],
            num_clusters=3,
            parameters={
                "clustering_model": {"agglomerative_clustering": {}},
                "projection_model": {"linear_discriminant_analysis": {}},
            },
        )

        assert r["status"] == "SUCCESS"
        assert "data" in r

    @pytest.mark.parametrize(
        "clustering_model,clustering_model_parameters",
        [
            [ClusteringModel.affinity_propagation, {}],
            [ClusteringModel.agglomerative_clustering, {}],
            [ClusteringModel.dbscan, {}],
            [ClusteringModel.dec, {"max_iteration": 5}],
            [ClusteringModel.kmeans, {}],
            [ClusteringModel.spectral_clustering, {}],
        ],
    )
    def test_segment_categorical_cols(
        self,
        clustering_task,
        clustering_model,
        clustering_model_parameters,
    ):
        df = pd.DataFrame({"x": ["a", "a", "c", "c", "c", "b", "b", "b"] * 3})

        r = clustering_task.run(
            df,
            num_clusters=3,
            parameters={
                "clustering_model": {clustering_model: clustering_model_parameters},
                "projection_model": {"linear_discriminant_analysis": {}},
            },
        )

        assert r["status"] == "SUCCESS"
        assert "data" in r

    @pytest.mark.parametrize(
        "clustering_model,clustering_model_parameters",
        [
            [ClusteringModel.affinity_propagation, {}],
            [ClusteringModel.agglomerative_clustering, {}],
            [ClusteringModel.dbscan, {}],
            [ClusteringModel.dec, {"max_iteration": 5}],
            [ClusteringModel.kmeans, {}],
            [ClusteringModel.spectral_clustering, {}],
        ],
    )
    @pytest.mark.parametrize(
        "project_model,projection_model_parameters",
        [
            [ProjectionModel.linear_discriminant_analysis, {}],
            [ProjectionModel.tsne, {}],
            [ProjectionModel.umap, {}],
        ],
    )
    def test_segment_mixed_type(
        self,
        clustering_task,
        clustering_model,
        clustering_model_parameters,
        project_model,
        projection_model_parameters,
    ):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, "b", "a"] * 2,
                "cat": ["a", "a", "c", "c", "c", "b", "b", "b", "a", "c"] * 2,
            }
        )

        r = clustering_task.run(
            df,
            num_clusters=3,
            parameters={
                "clustering_model": {clustering_model: clustering_model_parameters},
                "projection_model": {project_model: projection_model_parameters},
            },
        )

        assert r["status"] == "FAILURE"
        assert "validations" in r
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "DoNotContainMixedChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_insufficent_data(self, clustering_task):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10],
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            }
        )

        r = clustering_task.run(
            df,
            num_clusters=3,
            parameters={
                "clustering_model": {"agglomerative_clustering": {}},
                "projection_model": {"linear_discriminant_analysis": {}},
            },
        )

        assert r["status"] == "FAILURE"
        assert "validations" in r
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "IsSufficientDataChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_invalid_n_cluster_data(self, clustering_task):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        r = clustering_task.run(
            df,
            num_clusters=40,
            parameters={
                "clustering_model": {"agglomerative_clustering": {}},
                "projection_model": {"linear_discriminant_analysis": {}},
            },
        )

        assert r["status"] == "FAILURE"
        assert "validations" in r
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "IsValidNumberOfClusterChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_segment_with_explanations(self, clustering_task, init_ray):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        r = clustering_task.run(
            df,
            num_clusters=3,
            explain_samples=True,
            parameters={
                "clustering_model": {"dec": {"max_iteration": 2}},
                "projection_model": {"linear_discriminant_analysis": {}},
            },
        )

        assert r["status"] == "SUCCESS"
        assert "data" in r
        assert "data_v2" in r
        assert "shap_values" in r["data_v2"]

    def test_segment_empty_col(self, clustering_task):
        df = pd.DataFrame(
            {
                "x": [1, None, None, None, None, None, None, None, None, None] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        r = clustering_task.run(
            df,
            num_clusters=3,
            parameters={
                "clustering_model": {"agglomerative_clustering": {}},
                "projection_model": {"linear_discriminant_analysis": {}},
            },
        )

        assert r["status"] == "SUCCESS"
        assert "data" in r

    def test_max_train_samples(self, clustering_task):
        df = pd.DataFrame(
            {
                "x": [1, None, None, None, None, None, None, None, None, None] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        r = clustering_task.run(
            df,
            num_clusters=3,
            max_train_samples=10,
            parameters={
                "clustering_model": {"agglomerative_clustering": {}},
                "projection_model": {"linear_discriminant_analysis": {}},
            },
        )

        assert r["status"] == "SUCCESS"
        assert "data" in r

    def test_text_column(self, clustering_task):
        df = DatasetGenerator.generate(
            columns_parameters=[
                {"type": "text", "word_range": (5, 10)},
                {"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2},
            ],
            rows=20,
            random_state=0,
        )

        r = clustering_task.run(
            df,
            num_clusters=3,
            parameters={
                "clustering_model": {"agglomerative_clustering": {}},
                "projection_model": {"linear_discriminant_analysis": {}},
            },
        )

        assert r["status"] == "FAILURE"
        assert "validations" in r
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "DoNotContainTextChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL
