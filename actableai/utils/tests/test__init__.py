import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from actableai.utils import get_type_special_no_ag, is_fitted


def test_get_type_special_no_ag():
    result = get_type_special_no_ag(pd.Series([pd.NA for x in range(10)]))
    assert result != "text"
    assert result == "category"


def test_is_fitted():
    one_hot_encoder = OneHotEncoder()
    assert not is_fitted(one_hot_encoder)


def test_is_fitted_2():
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(np.array([[1, 2, 3]]))
    assert is_fitted(one_hot_encoder)
