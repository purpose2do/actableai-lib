import pandas as pd

from actableai.classification.utils import (
    leaderboard_cross_val,
    split_validation_by_datetime,
)


def test_leaderboard_cross_val():
    leaderboards = [
        pd.DataFrame(
            {
                "model": ["model1", "model2"],
                "score_val": [0, 1],
                "pred_time_val": [0, 1],
                "fit_time": [0, 1],
                "pred_time_val_marginal": [0, 1],
                "can_infer": [True, True],
                "fit_order": [0, 1],
                "fit_time_marginal": [0, 1],
            }
        ),
        pd.DataFrame(
            {
                "model": ["model1", "model2"],
                "score_val": [1, 2],
                "pred_time_val": [1, 2],
                "fit_time": [1, 2],
                "pred_time_val_marginal": [1, 2],
                "can_infer": [True, True],
                "fit_order": [1, 2],
                "fit_time_marginal": [1, 2],
            }
        ),
    ]
    leaderboard = leaderboard_cross_val(leaderboards)
    assert list(leaderboard["model"]) == ["model1", "model2"]
    assert list(leaderboard["score_val"]) == [0.5, 1.5]
    assert list(leaderboard["pred_time_val"]) == [0.5, 1.5]
    assert list(leaderboard["fit_time"]) == [0.5, 1.5]
    assert list(leaderboard["pred_time_val_marginal"]) == [0.5, 1.5]
    assert list(leaderboard["can_infer"]) == [True, True]
    assert list(leaderboard["fit_order"]) == [0.5, 1.5]
    assert list(leaderboard["fit_time_marginal"]) == [0.5, 1.5]
    assert "score_val_std" in list(leaderboard.columns)
    assert "pred_time_val_std" in list(leaderboard.columns)
    assert "fit_time_std" in list(leaderboard.columns)
    assert "pred_time_val_marginal_std" in list(leaderboard.columns)
    assert "can_infer_std" in list(leaderboard.columns)
    assert "fit_order_std" in list(leaderboard.columns)
    assert "fit_time_marginal_std" in list(leaderboard.columns)

    assert leaderboard is not None


def test_leaderboard_cross_val_reversed():
    leaderboards = [
        pd.DataFrame(
            {
                "model": ["model1", "model2"],
                "score_val": [0, 1],
                "pred_time_val": [0, 1],
                "fit_time": [0, 1],
                "pred_time_val_marginal": [0, 1],
                "can_infer": [True, True],
                "fit_order": [0, 1],
                "fit_time_marginal": [0, 1],
            }
        ),
        pd.DataFrame(
            {
                "model": ["model1", "model2"][::-1],
                "score_val": [1, 2][::-1],
                "pred_time_val": [1, 2][::-1],
                "fit_time": [1, 2][::-1],
                "pred_time_val_marginal": [1, 2][::-1],
                "can_infer": [True, True][::-1],
                "fit_order": [1, 2][::-1],
                "fit_time_marginal": [1, 2][::-1],
            }
        ),
    ]
    leaderboard = leaderboard_cross_val(leaderboards)
    assert list(leaderboard["model"]) == ["model1", "model2"]
    assert list(leaderboard["score_val"]) == [0.5, 1.5]
    assert list(leaderboard["pred_time_val"]) == [0.5, 1.5]
    assert list(leaderboard["fit_time"]) == [0.5, 1.5]
    assert list(leaderboard["pred_time_val_marginal"]) == [0.5, 1.5]
    assert list(leaderboard["can_infer"]) == [True, True]
    assert list(leaderboard["fit_order"]) == [0.5, 1.5]
    assert list(leaderboard["fit_time_marginal"]) == [0.5, 1.5]
    assert "score_val_std" in list(leaderboard.columns)
    assert "pred_time_val_std" in list(leaderboard.columns)
    assert "fit_time_std" in list(leaderboard.columns)
    assert "pred_time_val_marginal_std" in list(leaderboard.columns)
    assert "can_infer_std" in list(leaderboard.columns)
    assert "fit_order_std" in list(leaderboard.columns)
    assert "fit_time_marginal_std" in list(leaderboard.columns)


def test_split_validation_by_datetime():
    df_train = pd.DataFrame(
        {
            "datetime": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
            ],
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    df_train = df_train.sample(frac=1)
    val_ratio = 0.2
    df_train, df_val = split_validation_by_datetime(df_train, "datetime", val_ratio)
    assert df_train.shape[0] == 8
    assert df_val.shape[0] == 2
    assert not df_train.sort_values("datetime").equals(df_train)
    assert df_val.sort_values("datetime").equals(df_val)
