import pytest
import pandas as pd

from actableai import AAIAssociationRulesTask


@pytest.fixture(scope="function")
def association_rules_task():
    yield AAIAssociationRulesTask(use_ray=False)


class TestAssociationRules:
    def test_association_rules(
        self, association_rules_task: AAIAssociationRulesTask, tmp_path
    ):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, 2, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        result = association_rules_task.run(
            df=df,
            group_by=["x"],
            items="y",
        )

        assert result["status"] == "SUCCESS"
        assert "rules" in result["data"]
        assert result["data"]["rules"].shape == (10, 6)
        assert "frequent_itemset" in result["data"]
        assert result["data"]["frequent_itemset"].shape == (10, 2)
        assert "df_list" in result["data"]
        assert result["data"]["df_list"].shape == (10, 10)
