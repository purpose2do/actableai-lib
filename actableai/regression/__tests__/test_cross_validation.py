import pandas as pd
from tempfile import mkdtemp

from actableai.regression.cross_validation import run_cross_validation
from actableai.tasks.regression import _AAIRegressionTrainTask
from actableai.utils.testing import unittest_autogluon_hyperparameters


def test_cross_validation():
    task = _AAIRegressionTrainTask()
    df_train = pd.DataFrame(
        {"x": [x for x in range(0, 100)], "y": [y for y in range(100, 200)]}
    )
    df_test = pd.DataFrame(
        {"x": [x for x in range(0, 100)], "y": [y for y in range(100, 200)]}
    )
    (
        important_features,
        evaluate,
        predictions,
        predict_shap_values,
        df_val,
        leaderboard,
    ) = run_cross_validation(
        regression_train_task=task,
        kfolds=5,
        cross_validation_max_concurrency=1,
        explain_samples=False,
        presets="medium_quality_faster_train",
        hyperparameters=unittest_autogluon_hyperparameters(),
        model_directory=mkdtemp(prefix="autogluon_model"),
        target="y",
        features=["x"],
        run_model=False,
        df_train=df_train,
        df_test=df_test,
        prediction_quantiles=None,
        drop_duplicates=True,
        run_debiasing=False,
        biased_groups=[],
        debiased_features=[],
        residuals_hyperparameters=None,
        num_gpus=0,
        eval_metric="r2",
        time_limit=None,
        drop_unique=False,
        drop_useless_features=False,
        feature_prune=True,
        feature_prune_time_limit=None,
        num_trials=1,
        problem_type="regression",
        infer_limit=60,
        infer_limit_batch_size=100,
    )

    assert important_features is not None
    assert isinstance(important_features, list)
    assert len(important_features) == 1
    assert set([x["feature"] for x in important_features]) == set(["x"])
    assert "feature" in important_features[0]
    assert "importance" in important_features[0]
    assert "importance_std_err" in important_features[0]
    assert "p_value" in important_features[0]
    assert "p_value_std_err" in important_features[0]
    assert evaluate is not None
    assert df_val is not None
    assert leaderboard is not None
