import pandas as pd
import pytest

from actableai.utils.pdp_ice import get_pdp_and_ice
from actableai.utils.testing import (
    unittest_hyperparameters,
    unittest_autogluon_hyperparameters,
)

from actableai.tasks.regression import (
    AAIRegressionTask,
)
from actableai.tasks.classification import (
    AAIClassificationTask,
)


@pytest.fixture(scope="function")
def regression_task():
    yield AAIRegressionTask(use_ray=False)


@pytest.fixture(scope="function")
def classification_task():
    yield AAIClassificationTask(use_ray=False)


def test_pdp_ice_regression(regression_task, tmp_path):
    """
    Check if PDP and ICE for regression tasks runs without errors, and that
    the outputs are of the expected dimensions
    """

    df_train = pd.DataFrame(
        {
            "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            "z": ["a", "b", "b", "c", "a", "c", "a", "b", "a", "c"] * 2,
            "t": [2, 2, 2, 2, 2, 3, 3, 3, 4, 4] * 2,
        }
    )
    n_samples = len(df_train)

    result = regression_task.run(
        df=df_train,
        target="t",
        run_pdp=False,
        run_ice=False,
        model_directory=tmp_path,
        residuals_hyperparameters=unittest_autogluon_hyperparameters(),
        presets="medium_quality_faster_train",
        drop_duplicates=False,
        hyperparameters=unittest_hyperparameters(),
        drop_unique=False,
        drop_useless_features=False,
    )

    grid_resolution = 10
    feats = ["y", "z"]
    pd_r2 = get_pdp_and_ice(
        result["model"],
        df_train,
        features=feats,
        pdp=True,
        ice=True,
        grid_resolution=grid_resolution,
        verbosity=0,
        n_samples=None,
    )

    for feat_name in feats:
        n_unique = len(df_train[feat_name].unique())
        n_grid = min(n_unique, grid_resolution)
        assert pd_r2[feat_name]["individual"].shape == (1, n_samples, n_grid)
        assert pd_r2[feat_name]["average"].shape == (1, n_grid)
        assert pd_r2[feat_name]["values"][0].shape == (n_grid,)


def test_pdp_ice_regression_null(regression_task, tmp_path):
    """
    Check if PDP and ICE for regression tasks runs without errors, and that
    the outputs are of the expected dimensions

    In this case, each row contains at least one column with a null value;
    ensure that PDP/ICE can still be computed

    This also tests for handling of null values in numerical and categorical columns
    """

    df_train = pd.DataFrame(
        {
            "x": [None, 2, None, 2, 2, None, 3, None, 4, 4] * 2,
            "y": [1, None, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            "z": ["a", "b", "b", None, "a", "c", None, "b", None, "c"] * 2,
            "t": [2, 2, 2, 2, None, None, 3, 3, 4, None] * 2,
        }
    )
    n_samples = len(df_train)

    result = regression_task.run(
        df=df_train,
        target="t",
        run_pdp=False,
        run_ice=False,
        model_directory=tmp_path,
        residuals_hyperparameters=unittest_autogluon_hyperparameters(),
        presets="medium_quality_faster_train",
        drop_duplicates=False,
        hyperparameters=unittest_hyperparameters(),
        drop_unique=False,
        drop_useless_features=False,
    )

    grid_resolution = 10
    feats = ["x", "y", "z"]
    pd_r2 = get_pdp_and_ice(
        result["model"],
        df_train,
        features=feats,
        pdp=True,
        ice=True,
        grid_resolution=grid_resolution,
        verbosity=0,
        n_samples=None,
    )

    for feat_name in feats:
        df_train_drop = df_train.dropna(subset=[feat_name])

        n_unique = len(df_train_drop[feat_name].unique())
        n_grid = min(n_unique, grid_resolution)
        n_samples_drop = len(df_train_drop)
        assert pd_r2[feat_name]["individual"].shape == (1, n_samples_drop, n_grid)
        assert pd_r2[feat_name]["average"].shape == (1, n_grid)
        assert pd_r2[feat_name]["values"][0].shape == (n_grid,)


def test_pdp_ice_classification(classification_task, tmp_path):
    """
    Check if PDP and ICE for classification tasks runs without errors, and that
    the outputs are of the expected dimensions
    """

    df_train = pd.DataFrame(
        {
            "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 3,
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
            "z": ["a", "b", "b", "c", "a", "c", "a", "b", "a", "c"] * 3,
            "t": [2, 2, 2, 2, 2, 3, 3, 3, 4, 4] * 3,
        }
    )
    n_samples = len(df_train)

    result = classification_task.run(
        df=df_train,
        target="t",
        run_pdp=False,
        run_ice=False,
        model_directory=tmp_path,
        residuals_hyperparameters=unittest_autogluon_hyperparameters(),
        presets="medium_quality_faster_train",
        drop_duplicates=False,
        hyperparameters=unittest_autogluon_hyperparameters(),
        drop_unique=False,
        drop_useless_features=False,
    )

    grid_resolution = 10
    feats = ["y", "z"]

    pd_r2 = get_pdp_and_ice(
        result["model"],
        df_train,
        features=feats,
        pdp=True,
        ice=True,
        grid_resolution=grid_resolution,
        verbosity=0,
        n_samples=None,
    )

    n_unique_target = len(df_train["t"].unique())
    for feat_name in feats:
        n_unique = len(df_train[feat_name].unique())
        n_grid = min(n_unique, grid_resolution)
        assert pd_r2[feat_name]["individual"].shape == (
            n_unique_target,
            n_samples,
            n_grid,
        )
        assert pd_r2[feat_name]["average"].shape == (n_unique_target, n_grid)
        assert pd_r2[feat_name]["values"][0].shape == (n_grid,)


def test_pdp_ice_classification_null(classification_task, tmp_path):
    """
    Check if PDP and ICE for classification tasks runs without errors, and that
    the outputs are of the expected dimensions

    In this case, each row contains at least one column with a null value;
    ensure that PDP/ICE can still be computed

    This also tests for handling of null values in numerical and categorical columns
    """

    df_train = pd.DataFrame(
        {
            "x": [None, 2, None, 2, None, None, 3, 3, None, 4] * 5,
            "y": [1, None, 3, 4, 5, None, 7, None, 9, 10] * 5,
            "z": ["a", "b", None, None, "a", "c", None, "b", "a", None] * 5,
            "t": [2, 2, 2, 2, 2, 3, 3, 3, 4, 4] * 5,
        }
    )
    n_samples = len(df_train)

    result = classification_task.run(
        df=df_train,
        target="t",
        run_pdp=False,
        run_ice=False,
        model_directory=tmp_path,
        residuals_hyperparameters=unittest_autogluon_hyperparameters(),
        presets="medium_quality_faster_train",
        drop_duplicates=False,
        hyperparameters=unittest_autogluon_hyperparameters(),
        drop_unique=False,
        drop_useless_features=False,
    )

    grid_resolution = 10
    feats = ["y", "z"]

    pd_r2 = get_pdp_and_ice(
        result["model"],
        df_train,
        features=feats,
        pdp=True,
        ice=True,
        grid_resolution=grid_resolution,
        verbosity=0,
        n_samples=None,
    )

    for feat_name in feats:
        df_train_drop = df_train.dropna(subset=[feat_name])
        n_unique_target = len(df_train_drop["t"].unique())

        n_unique = len(df_train_drop[feat_name].unique())
        n_grid = min(n_unique, grid_resolution)
        n_samples_drop = len(df_train_drop)
        assert pd_r2[feat_name]["individual"].shape == (
            n_unique_target,
            n_samples_drop,
            n_grid,
        )
        assert pd_r2[feat_name]["average"].shape == (n_unique_target, n_grid)
        assert pd_r2[feat_name]["values"][0].shape == (n_grid,)
