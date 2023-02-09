import pytest
import pandas as pd

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
def payload():
    return PCPayload(
        dataset=Dataset(
            data=pd.DataFrame(
                data=[
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20],
                    [21, 22, 23, 24, 25],
                ],
                columns=["A", "B", "C", "D", "E"],
            ).to_dict(orient="list")
        ),
        constraints=Constraints(causes=[], effects=[], forbiddenRelationships=[]),
        causal_variables=[
            CausalVariable(name="A"),
            CausalVariable(name="B"),
            CausalVariable(name="C"),
            CausalVariable(name="D"),
            CausalVariable(name="E"),
        ],
        model_save_dir="/tmp",
    )


class TestCausalDiscoveryTask:
    def test_run(self, payload):
        task = AAICausalDiscoveryTask()
        graph = task.run("pc", payload)
        assert type(graph) is dict

    def test_invalid_algorithm(self, payload):
        task = AAICausalDiscoveryTask()
        with pytest.raises(ValueError):
            task.run("invalid", payload)
