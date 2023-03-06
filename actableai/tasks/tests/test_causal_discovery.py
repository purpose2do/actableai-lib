import pytest
import pandas as pd

from actableai.causal.discover.algorithms.direct_lingam import DirectLiNGAMPayload
from actableai.causal.discover.algorithms.notears import NotearsPayload
from actableai.data_validation.base import CheckLevels
from actableai.tasks.causal_discovery import AAICausalDiscoveryTask
from actableai.causal.discover.model.causal_discovery import (
    Dataset,
    Constraints,
    CausalVariable,
)
from actableai.causal.discover.algorithms.commons.base_runner import CausalGraph
from actableai.causal.discover.algorithms.deci import DeciPayload
from actableai.causal.discover.algorithms.pc import PCPayload


@pytest.fixture(scope="function")
def causal_discovery_task():
    yield AAICausalDiscoveryTask(use_ray=False)


class TestCausalDiscoveryTask:
    @staticmethod
    def _get_payload(algo, model_path):
        payload = {
            "dataset": {
                "data": pd.DataFrame(
                    data=[
                        [1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 10],
                        [11, 12, 13, 14, 15],
                        [16, 17, 18, 19, 20],
                        [21, 22, 23, 24, 25],
                        [26, 27, 28, 29, 30],
                    ],
                    columns=["A", "B", "C", "D", "E"],
                ).to_dict(orient="list"),
            },
            "constraints": {
                "causes": [],
                "effects": [],
                "forbiddenRelationships": [],
            },
            "causal_variables": [
                {"name": "A"},
                {"name": "B"},
                {"name": "C"},
                {"name": "D"},
                {"name": "E"},
            ],
            "model_save_dir": str(model_path),
        }

        if algo == "deci":
            payload["training_options"] = {
                "max_auglag_inner_epochs": 10,
            }
            return DeciPayload(**payload)
        if algo == "notears":
            return NotearsPayload(**payload)
        if algo == "direct-lingam":
            return DirectLiNGAMPayload(**payload)
        if algo == "pc":
            return PCPayload(**payload)

    @pytest.mark.parametrize(
        "algo",
        [
            "deci",
            "notears",
            "direct-lingam",
            "pc",
        ],
    )
    def test_run_success(self, causal_discovery_task, algo, tmp_path):
        payload = self._get_payload(algo=algo, model_path=tmp_path)

        results = causal_discovery_task.run(
            algo=algo,
            payload=payload,
        )

        assert results is not None
        assert "status" in results
        assert results["status"] == "SUCCESS"
        assert "data" in results
        assert type(results["data"]["causal_graph"]) is dict

        graph = results["data"]["causal_graph"]
        assert "directed" in graph
        assert graph["directed"] == True
        assert "elements" in graph
        graph_elements = graph["elements"]
        assert "nodes" in graph_elements
        for node in graph_elements["nodes"]:
            assert "data" in node
            node_data = node["data"]
            assert "id" in node_data
            assert "value" in node_data
            assert "name" in node_data
        assert "edges" in graph_elements
        for edge in graph_elements["edges"]:
            assert "data" in edge
            edge_data = edge["data"]
            assert "source" in edge_data
            assert "target" in edge_data

    def test_invalid_algorithm(self, causal_discovery_task, tmp_path):
        payload = self._get_payload(algo="direct-lingam", model_path=tmp_path)

        results = causal_discovery_task.run(
            algo="invalid",
            payload=payload,
        )

        assert results is not None
        assert "status" in results
        assert results["status"] == "FAILURE"

        validations_dict = {val["name"]: val["level"] for val in results["validations"]}

        assert "CausalDiscoveryAlgoChecker" in validations_dict
        assert validations_dict["CausalDiscoveryAlgoChecker"] == CheckLevels.CRITICAL
