from actableai.data_imputation.utils.list_utils import all_possible_pairs


def test_all_possible_pairs():
    a = {1, 2}
    b = {3, 4}
    c = {5, 6}
    assert all_possible_pairs(a, b, c) == [
        [1, 3, 5],
        [1, 3, 6],
        [1, 4, 5],
        [1, 4, 6],
        [2, 3, 5],
        [2, 3, 6],
        [2, 4, 5],
        [2, 4, 6],
    ]
