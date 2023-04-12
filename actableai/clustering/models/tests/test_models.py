from typing import Dict, Any

import pytest

import pandas as pd

from actableai.clustering.models import model_dict, Model


class TestClusteringModel:
    @pytest.mark.parametrize("num_clusters", [2, 6])
    @pytest.mark.parametrize(
        "model_type,parameters",
        [
            [Model.affinity_propagation, {}],
            [Model.agglomerative_clustering, {}],
            [Model.dbscan, {}],
            [Model.dec, {"max_iteration": 5}],
            [Model.kmeans, {}],
            [Model.spectral_clustering, {}],
        ],
    )
    def test_simple_model(
        self,
        num_clusters: int,
        model_type: Model,
        parameters: Dict[str, Any],
    ):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, -1, -2, 8, 9, 10] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        model = model_dict[model_type](
            input_size=df.shape[-1],
            num_clusters=num_clusters,
            parameters=parameters,
            process_parameters=True,
            verbosity=0,
        )

        data = df.to_numpy()

        model.fit(data)
        prediction = model.predict(data)

        assert prediction is not None
        assert prediction.shape[0] == data.shape[0]
