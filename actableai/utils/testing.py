import datetime
import itertools
import numpy as np
import pandas as pd
import random
import ray

from actableai.timeseries.dataset import AAITimeSeriesDataset


def unittest_autogluon_hyperparameters():
    return {"RF": {}}


def unittest_hyperparameters():
    return {"rf": {}}


def init_ray(**kwargs):
    ray.init(num_cpus=1, namespace="aai", **kwargs)


def generate_random_date(
    np_rng=None,
    min_year=1900,
    min_month=1,
    min_day=1,
    max_year=2000,
    max_month=1,
    max_day=1,
    random_state=None,
):
    """
    TODO write documentation
    """
    min_date = datetime.date(min_year, min_month, min_day)
    max_date = datetime.date(max_year, max_month, max_day)

    max_days = (max_date - min_date).days

    days_delta = None
    if np_rng is None:
        days_delta = random.randrange(max_days)
    else:
        days_delta = np_rng.integers(max_days)

    return min_date + datetime.timedelta(days=int(days_delta))


def generate_date_range(
    np_rng=None,
    start_date=None,
    min_periods=10,
    max_periods=60,
    periods=None,
    freq=None,
):
    """
    TODO write documentation
    """

    if start_date is None:
        start_date = generate_random_date(np_rng)
    if periods is None:
        if np_rng is None:
            periods = random.randrange(min_periods, max_periods)
        else:
            periods = np_rng.integers(min_periods, max_periods)
    if freq is None:
        freq = "T"

    return pd.date_range(start_date, periods=periods, freq=freq)


def generate_forecast_dataset(
    np_rng,
    prediction_length,
    n_groups=1,
    n_targets=1,
    freq=None,
    n_real_dynamic_features=0,
    n_cat_dynamic_features=0,
    n_real_static_features=0,
    n_cat_static_features=0,
    date_range_kwargs=None,
):
    """
    TODO write documentation
    """
    if date_range_kwargs is None:
        date_range_kwargs = {}

    df_dict = {}
    target_list = [f"target_{i}" for i in range(n_targets)]
    feat_dynamic_real = [f"real_feat_{i}" for i in range(n_real_dynamic_features)]
    feat_dynamic_cat = [f"cat_feat_{i}" for i in range(n_cat_dynamic_features)]
    feat_static_real = [f"real_static_feat_{i}" for i in range(n_real_static_features)]
    feat_static_cat = [f"cat_static_feat_{i}" for i in range(n_cat_static_features)]

    for i in range(n_groups):
        date_range = generate_date_range(np_rng, freq=freq, **date_range_kwargs)

        data = {}
        for target in target_list:
            data[target] = np_rng.standard_normal(len(date_range))
        for feat in feat_dynamic_real:
            data[feat] = np_rng.standard_normal(len(date_range))
        for feat in feat_dynamic_cat:
            data[feat] = np_rng.integers(1, 10, len(date_range))
        for feat in feat_static_real:
            data[feat] = np_rng.standard_normal()
        for feat in feat_static_cat:
            data[feat] = np_rng.integers(1, 10)

        df = pd.DataFrame(data)
        df.index = date_range

        df_dict[f"group_{i}"] = df

    return AAITimeSeriesDataset(
        dataframes=df_dict,
        target_columns=target_list,
        freq=freq,
        prediction_length=prediction_length,
        feat_dynamic_real=feat_dynamic_real,
        feat_dynamic_cat=feat_dynamic_cat,
        feat_static_real=feat_static_real,
        feat_static_cat=feat_static_cat,
    )


def generate_forecast_df(
    np_rng,
    prediction_length,
    n_group_by=0,
    n_targets=1,
    freq=None,
    n_real_static_features=0,
    n_cat_static_features=0,
    n_real_dynamic_features=0,
    n_cat_dynamic_features=0,
    date_range_kwargs=None,
):
    """
    TODO write documentation
    """
    has_groups = n_group_by > 0
    group_values_list = []

    n_groups = 1
    if has_groups:
        for group_by_index in range(n_group_by):
            n_group_cat = np_rng.integers(2, 5)
            n_groups *= n_group_cat

            group_values_list.append([f"group_val_{i}" for i in range(n_group_cat)])

    group_by = [f"group_{i}" for i in range(n_group_by)]

    dataset = generate_forecast_dataset(
        np_rng,
        prediction_length=prediction_length,
        n_groups=n_groups,
        n_targets=n_targets,
        freq=freq,
        n_real_dynamic_features=n_real_dynamic_features,
        n_cat_dynamic_features=n_cat_dynamic_features,
        n_real_static_features=n_real_static_features,
        n_cat_static_features=n_cat_static_features,
        date_range_kwargs=date_range_kwargs,
    )

    feature_list = (
        dataset.feat_dynamic_real
        + dataset.feat_dynamic_cat
        + dataset.feat_static_real
        + dataset.feat_static_cat
    )

    df = pd.DataFrame()

    for (group, df_group), group_values in zip(
        dataset.dataframes.items(), itertools.product(*group_values_list)
    ):
        df_group["_date"] = df_group.index
        df_group["_date_str"] = df_group["_date"].astype(str)

        if has_groups:
            for group_by_name, group_by_value in zip(group_by, group_values):
                df_group[group_by_name] = group_by_value

        if len(feature_list) > 0:
            df_group.loc[-prediction_length:, dataset.target_columns] = np.nan

        df = pd.concat([df, df_group], ignore_index=True)

    return (
        df,
        "_date",
        "_date_str",
        dataset.target_columns,
        group_by,
        feature_list,
        n_groups,
    )
