from typing import List, Optional, Dict
import pandas as pd
import pytest

from actableai import AAIInterventionTask


@pytest.fixture(scope="function")
def intervention_task():
    yield AAIInterventionTask(use_ray=False)


class TestIntervention:
    def test_intervention_numeric(
        self, intervention_task: AAIInterventionTask, tmp_path
    ):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "current_intervention": [2, 2, 2, 2, None, 3, None, 3, 4, 4] * 2,
            }
        )
        df["new_intervention"] = df["current_intervention"] * 2

        result = intervention_task.run(
            df,
            "y",
            "current_intervention",
            "new_intervention",
            model_directory=tmp_path,
        )

        assert result["status"] == "SUCCESS"
        assert "df" in result
        assert result["df"].shape == (20, 6)

    def test_intervention_numeric_cate(
        self, intervention_task: AAIInterventionTask, tmp_path
    ):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "current_intervention": [2, 2, 2, 2, None, 3, None, 3, 4, 4] * 2,
            }
        )
        df["new_intervention"] = df["current_intervention"] * 2

        result = intervention_task.run(
            df,
            "y",
            "current_intervention",
            "new_intervention",
            model_directory=tmp_path,
            cate_alpha=0.5,
        )

        assert result["status"] == "SUCCESS"
        assert "df" in result
        assert result["df"].shape == (20, 10)

    def test_intervention_categorical(
        self, intervention_task: AAIInterventionTask, tmp_path
    ):
        current_intervention = ["a", None, "a", "a", "a", "b", "b", "b", "b", "b"]
        new_intervention = ["b", "b", "b", "b", "b", None, None, "a", "a", "a"]
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "current_intervention": current_intervention * 2,
                "new_intervention": new_intervention * 2,
            }
        )

        r = intervention_task.run(
            df,
            "y",
            "current_intervention",
            "new_intervention",
            model_directory=tmp_path,
        )

        assert r["status"] == "SUCCESS"
        assert "df" in r
        assert r["df"].shape == (20, 6)

    def test_intervention_categorical_cate(
        self, intervention_task: AAIInterventionTask, tmp_path
    ):
        current_intervention = ["a", None, "a", "a", "a", "b", "b", "b", "b", "b"]
        new_intervention = ["b", "b", "b", "b", "b", None, None, "a", "a", "a"]
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "current_intervention": current_intervention * 2,
                "new_intervention": new_intervention * 2,
            }
        )

        r = intervention_task.run(
            df,
            "y",
            "current_intervention",
            "new_intervention",
            model_directory=tmp_path,
            cate_alpha=0.5,
        )

        assert r["status"] == "SUCCESS"
        assert "df" in r
        assert r["df"].shape == (20, 10)

    def test_intervention_with_common_causes(
        self, intervention_task, tmp_path
    ):
        current_intervention = ["a", None, "a", "a", "a", "b", "b", "b", "b", "b"]
        new_intervention = ["b", "b", "b", "b", "b", None, None, "a", "a", "a"]
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": ["a", "a", "a", "b", "b", "b", "b", "b", "b", "b"] * 2,
                "current_intervention": current_intervention * 2,
                "new_intervention": new_intervention * 2,
            }
        )

        r = intervention_task.run(
            df,
            "y",
            "current_intervention",
            "new_intervention",
            common_causes=["y", "z"],
            model_directory=tmp_path,
        )

        assert r["status"] == "SUCCESS"
        assert "df" in r
        assert r["df"].shape == (20, 11)

    def test_intervention_with_common_causes_cate(
        self, intervention_task, tmp_path
    ):
        current_intervention = ["a", None, "a", "a", "a", "b", "b", "b", "b", "b"]
        new_intervention = ["b", "b", "b", "b", "b", None, None, "a", "a", "a"]
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": ["a", "a", "a", "b", "b", "b", "b", "b", "b", "b"] * 2,
                "current_intervention": current_intervention * 2,
                "new_intervention": new_intervention * 2,
            }
        )

        r = intervention_task.run(
            df,
            "y",
            "current_intervention",
            "new_intervention",
            common_causes=["y", "z"],
            model_directory=tmp_path,
            cate_alpha=0.5,
        )

        assert r["status"] == "SUCCESS"
        assert "df" in r
        assert r["df"].shape == (20, 11)
