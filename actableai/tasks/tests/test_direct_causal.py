import pandas as pd

from actableai.tasks.direct_causal import AAIDirectCausalFeatureSelection


class TestCausalFeatureSelection:
    def test_mixed_dataset(self):
        df = pd.DataFrame(
            {
                "x": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"] * 10,
                "w": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"] * 10,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] * 10,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] * 10,
            }
        )

        task = AAIDirectCausalFeatureSelection()
        re = task.run(df, "y", ["x", "w", "z"])
        print(re)
        assert re is not None
        assert re["status"] == "SUCCESS"
