from typing import Dict, Any

import pytest

import pandas as pd

from actableai.embedding.models import model_dict, Model


class TestEmbeddingModel:
    @pytest.mark.parametrize("embedding_size", [1, 2])
    @pytest.mark.parametrize(
        "model_type,parameters",
        [
            [Model.linear_discriminant_analysis, {}],
            [Model.tsne, {}],
            [Model.umap, {}],
        ],
    )
    def test_simple_model(
        self,
        embedding_size: int,
        model_type: Model,
        parameters: Dict[str, Any],
    ):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, -1, -2, 8, 9, 10] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "target": [1, 3, 1, 0, 1, 0, 1, 0, 0, 1] * 2,
            }
        )

        model = model_dict[model_type](
            embedding_size=embedding_size,
            parameters=parameters,
            process_parameters=True,
            verbosity=0,
        )

        data = df[["x", "y", "z"]].to_numpy()
        target = df["target"].to_numpy()

        model.fit(data, target)
        embedding = model.transform(data)

        assert embedding is not None
        assert embedding.shape[0] == data.shape[0]
        assert embedding.shape[1] == embedding_size
