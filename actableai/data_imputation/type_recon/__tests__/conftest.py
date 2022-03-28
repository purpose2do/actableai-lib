import os

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def manual_data():
    return pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "experiments/data/manual_test/all_types.csv",
        )
    )
