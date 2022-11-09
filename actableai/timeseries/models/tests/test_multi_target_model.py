import mxnet as mx
import pandas as pd
import pytest
import ray

from actableai.exceptions.timeseries import UntrainedModelException
from actableai.timeseries.models import params
from actableai.timeseries.models.independent_multivariate_model import (
    AAITimeSeriesIndependentMultivariateModel,
)
from actableai.utils.testing import init_ray, generate_forecast_dataset


@pytest.fixture(scope="module")
def mx_ctx():
    yield mx.cpu()


class TestAAITimeSeriesMultiTargetModel:
    def _fit_predict_model(
        self,
        mx_ctx,
        prediction_length,
        model_params,
        train_dataset=None,
        valid_dataset=None,
        test_dataset=None,
        trials=1,
        use_ray=False,
    ):
        model = AAITimeSeriesIndependentMultivariateModel(
            prediction_length=prediction_length
        )

        ray_shutdown = False
        if use_ray and not ray.is_initialized():
            init_ray()
            ray_shutdown = True

        if train_dataset is not None:
            model.fit(
                dataset=train_dataset,
                model_params=model_params,
                mx_ctx=mx_ctx,
                loss="mean_wQuantileLoss",
                trials=trials,
                max_concurrent=4 if not use_ray else 1,
                use_ray=use_ray,
                tune_samples=3,
                sampling_method="random",
                random_state=0,
                ray_tune_kwargs={
                    "resources_per_trial": {
                        "cpu": 1,
                        "gpu": 0,
                    },
                    "raise_on_failed_trial": False,
                },
                verbose=3,
            )

        if ray_shutdown:
            ray.shutdown()

        validations = None
        if valid_dataset is not None:
            df_val_predictions_dict, df_item_metrics_dict, df_agg_metrics = model.score(
                dataset=valid_dataset
            )
            validations = {
                "predictions": df_val_predictions_dict,
                "item_metrics": df_item_metrics_dict,
                "agg_metrics": df_agg_metrics,
            }

            if test_dataset is not None:
                model.refit(dataset=valid_dataset)

        predictions = None
        if test_dataset is not None:
            df_predictions_dict = model.predict(dataset=test_dataset)
            predictions = {"predictions": df_predictions_dict}

        return validations, predictions

    @pytest.mark.parametrize("n_targets", [1, 2])
    @pytest.mark.parametrize("n_groups", [1, 2])
    @pytest.mark.parametrize("use_features", [True, False])
    @pytest.mark.parametrize("freq", ["T", "MS", "YS"])
    def test_simple_model(
        self,
        np_rng,
        mx_ctx,
        n_targets,
        n_groups,
        use_features,
        freq,
    ):
        prediction_length = np_rng.integers(1, 3)
        dataset = generate_forecast_dataset(
            np_rng,
            prediction_length=prediction_length,
            n_groups=n_groups,
            n_targets=n_targets,
            freq=freq,
            n_real_dynamic_features=np_rng.integers(1, 10) if use_features else 0,
            n_cat_dynamic_features=np_rng.integers(1, 10) if use_features else 0,
            n_real_static_features=np_rng.integers(2, 10) if use_features else 0,
            n_cat_static_features=np_rng.integers(2, 10) if use_features else 0,
        )
        group_list = dataset.group_list

        slice_last_valid_index = lambda df: slice(
            -prediction_length if use_features else df.shape[0]
        )

        test_dataset = dataset
        valid_dataset = test_dataset.slice_data(slice_df=slice_last_valid_index)
        train_dataset = valid_dataset.slice_data(slice_df=slice(-prediction_length))

        model_params = [
            params.ConstantValueParams(),
            params.MultivariateConstantValueParams(),
        ]

        validations, predictions = self._fit_predict_model(
            mx_ctx=mx_ctx,
            prediction_length=prediction_length,
            model_params=model_params,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
        )

        assert validations is not None
        assert "predictions" in validations
        assert "item_metrics" in validations
        assert "agg_metrics" in validations
        assert predictions is not None
        assert "predictions" in predictions

        df_predictions_dict = validations["predictions"]
        for group in df_predictions_dict.keys():
            assert group in group_list

        df_item_metrics_dict = validations["item_metrics"]
        for group in df_item_metrics_dict.keys():
            assert group in group_list

        for group in group_list:
            df_predictions = df_predictions_dict[group]
            assert "target" in df_predictions.columns
            assert "date" in df_predictions.columns
            assert "0.05" in df_predictions.columns
            assert "0.5" in df_predictions.columns
            assert "0.95" in df_predictions.columns
            assert len(df_predictions) == prediction_length * n_targets
            assert (
                df_predictions.groupby("date").first().index
                == valid_dataset.dataframes[group].index[-prediction_length:]
            ).all()

            df_item_metrics = df_item_metrics_dict[group]
            assert len(df_item_metrics) == n_targets
            assert "target" in df_item_metrics.columns
            assert "MAPE" in df_item_metrics.columns
            assert "MASE" in df_item_metrics.columns
            assert "RMSE" in df_item_metrics.columns
            assert "sMAPE" in df_item_metrics.columns
            assert (
                not df_item_metrics[["MAPE", "MASE", "RMSE", "sMAPE"]]
                .isna()
                .any(axis=None)
            )

        df_agg_metrics = validations["agg_metrics"]
        assert len(df_agg_metrics) == n_targets
        assert "target" in df_agg_metrics.columns
        assert "MAPE" in df_agg_metrics.columns
        assert "MASE" in df_agg_metrics.columns
        assert "RMSE" in df_agg_metrics.columns
        assert "sMAPE" in df_agg_metrics.columns
        assert (
            not df_agg_metrics[["MAPE", "MASE", "RMSE", "sMAPE"]].isna().any(axis=None)
        )

        df_predictions_dict = predictions["predictions"]
        for group in df_predictions_dict.keys():
            assert group in group_list

        for group in group_list:
            if not dataset.has_dynamic_features:
                future_dates = pd.date_range(
                    test_dataset.dataframes[group].index[-1],
                    periods=prediction_length + 1,
                    freq=freq,
                )[1:]
            else:
                future_dates = test_dataset.dataframes[group].index[-prediction_length:]

            df_predictions = df_predictions_dict[group]
            assert "target" in df_predictions.columns
            assert "date" in df_predictions.columns
            assert "0.05" in df_predictions.columns
            assert "0.5" in df_predictions.columns
            assert "0.95" in df_predictions.columns
            assert len(df_predictions) == prediction_length * n_targets
            assert (df_predictions.groupby("date").first().index == future_dates).all()

    @pytest.mark.parametrize("n_targets", [1, 2])
    @pytest.mark.parametrize("freq", ["T", "MS", "YS"])
    @pytest.mark.parametrize("use_ray", [True, False])
    def test_hyperopt(self, np_rng, mx_ctx, use_ray, freq, n_targets):
        prediction_length = np_rng.integers(1, 3)
        dataset = generate_forecast_dataset(
            np_rng,
            prediction_length=prediction_length,
            n_groups=1,
            n_targets=n_targets,
            freq=freq,
        )

        slice_last_valid_index = lambda df: slice(df.shape[0])

        test_dataset = dataset
        valid_dataset = test_dataset.slice_data(slice_df=slice_last_valid_index)
        train_dataset = valid_dataset.slice_data(slice_df=slice(-prediction_length))

        model_params = [
            params.ConstantValueParams(),
            params.MultivariateConstantValueParams(),
        ]

        validations, predictions = self._fit_predict_model(
            mx_ctx=mx_ctx,
            prediction_length=prediction_length,
            model_params=model_params,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            trials=10,
            use_ray=use_ray,
        )

        assert validations is not None
        assert "predictions" in validations
        assert "item_metrics" in validations
        assert "agg_metrics" in validations
        assert predictions is not None
        assert "predictions" in predictions

    @pytest.mark.parametrize("freq", ["T"])
    def test_not_trained_score(self, np_rng, mx_ctx, freq):
        prediction_length = np_rng.integers(1, 3)
        dataset = generate_forecast_dataset(
            np_rng,
            prediction_length=prediction_length,
            n_groups=1,
            n_targets=1,
            freq=freq,
        )

        model_params = [params.ConstantValueParams()]

        with pytest.raises(UntrainedModelException):
            _, _ = self._fit_predict_model(
                mx_ctx=mx_ctx,
                prediction_length=prediction_length,
                model_params=model_params,
                train_dataset=None,
                valid_dataset=dataset,
                trials=1,
                use_ray=False,
            )

    @pytest.mark.parametrize("freq", ["T"])
    def test_not_trained_predict(self, np_rng, mx_ctx, freq):
        prediction_length = np_rng.integers(1, 3)
        dataset = generate_forecast_dataset(
            np_rng,
            prediction_length=prediction_length,
            n_groups=1,
            n_targets=1,
            freq=freq,
        )

        model_params = [params.ConstantValueParams()]

        with pytest.raises(UntrainedModelException):
            _, _ = self._fit_predict_model(
                mx_ctx=mx_ctx,
                prediction_length=prediction_length,
                model_params=model_params,
                train_dataset=None,
                test_dataset=dataset,
                trials=1,
                use_ray=False,
            )
