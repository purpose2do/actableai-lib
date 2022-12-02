import numpy as np
import pandas as pd
import pytest

from actableai.timeseries.utils import (
    find_freq,
    handle_datetime_column,
)
from actableai.utils.testing import generate_date_range


# Test find_freq
@pytest.mark.parametrize("freq", ["H", "S", "T", "N"])
@pytest.mark.parametrize("freq_n", ["", "15", "2", "30"])
def test_find_freq(np_rng, freq, freq_n):
    freq = freq_n + freq
    pd_date = pd.Series(generate_date_range(np_rng, freq=freq))
    assert find_freq(pd_date) == freq


@pytest.mark.parametrize("freq", ["H", "S", "T", "N"])
@pytest.mark.parametrize("freq_n", ["", "15", "2", "30"])
def test_find_freq_missing_values(np_rng, freq, freq_n):
    freq = freq_n + freq
    pd_date = pd.Series(generate_date_range(np_rng, periods=10, freq=freq))
    # Remove two values in the ten values
    pd_date[3] = np.nan
    pd_date[7] = np.nan
    assert find_freq(pd_date, period=3) == freq


def test_find_freq_not_enough_values(np_rng):
    pd_date = pd.Series(generate_date_range(np_rng, periods=2))
    assert find_freq(pd_date) is None


@pytest.mark.parametrize(
    "start_date",
    ["2021-02-06", "2015-09-08", "18:00", "18:30:25", "2020-01-02 05:00:00+02:00"],
)
@pytest.mark.parametrize("freq", ["T", "H", "S", "Y", "us"])
def test_handle_datetime_column_pd_datetime(start_date, freq):
    date_range = pd.Series(pd.date_range(start_date, periods=45, freq=freq))
    parsed_dt, dtype = handle_datetime_column(date_range)

    assert parsed_dt is not None
    assert dtype == "datetime"
    assert (parsed_dt == date_range).all()


@pytest.mark.parametrize(
    "start_date", ["2021-02-06", "2015-09-08", "2020-01-02 05:00:00+02:00"]
)
@pytest.mark.parametrize("freq", ["T", "H", "S", "Y", "us"])
def test_handle_datetime_column_str(start_date, freq):
    date_range = pd.Series(pd.date_range(start_date, periods=10, freq=freq))
    parsed_dt, dtype = handle_datetime_column(date_range.astype(str))

    assert parsed_dt is not None
    assert dtype == "datetime"
    for i in range(len(date_range)):
        revert_date = None
        if date_range[i].day <= 12:
            revert_date = date_range[i].replace(
                month=date_range[i].day, day=date_range[i].month
            )
        assert parsed_dt[i] == date_range[i] or parsed_dt[i] == revert_date


def test_handle_datetime_column_mixed_hour():
    date_range = pd.Series(pd.date_range("18:00", periods=10, freq="H"))
    parsed_dt, dtype = handle_datetime_column(date_range.astype(str))

    assert parsed_dt is not None
    assert dtype == "datetime"
    assert (parsed_dt.dt.time == date_range.dt.time).all()


@pytest.mark.parametrize("freq", ["T", "S", "Y", "us"])
def test_handle_datetime_column_hour_multi_freq(freq):
    date_range = pd.Series(pd.date_range("18:00", periods=10, freq=freq))
    parsed_dt, dtype = handle_datetime_column(date_range.astype(str))

    assert parsed_dt is not None
    assert dtype == "datetime"
    assert (parsed_dt.dt.time == date_range.dt.time).all()


def test_handle_datetime_column_tstamp_mixed():
    date_range = pd.Series(pd.date_range("18:30:25", periods=10, freq="H"))
    parsed_dt, dtype = handle_datetime_column(date_range.astype(str))

    assert parsed_dt is not None
    assert dtype == "datetime"
    assert (parsed_dt.dt.time == date_range.dt.time).all()


@pytest.mark.parametrize("freq", ["T", "S", "Y", "us"])
def test_handle_datetime_column_tstamp_multi_freq(freq):
    date_range = pd.Series(pd.date_range("18:30:25", periods=10, freq=freq))
    parsed_dt, dtype = handle_datetime_column(date_range.astype(str))

    assert parsed_dt is not None
    assert dtype == "datetime"
    assert (parsed_dt.dt.time == date_range.dt.time).all()


def test_handle_datetime_column_dotted():
    date_range = pd.Series(["{}.{}.20{:0>2}".format(i, i, i) for i in range(1, 10)])
    parsed_dt, dtype = handle_datetime_column(date_range)

    assert parsed_dt is not None
    assert dtype == "datetime"
    for i in range(len(date_range)):
        assert parsed_dt[i] == pd.Timestamp(date_range[i])


def test_handle_datetime_column_nan_values_threshold():
    date_range = pd.Series(["{}.{}.20{:0>2}".format(i, i, i) for i in range(1, 10)])
    date_range[0:3] = ["abc" for _ in range(3)]
    parsed_dt, dtype = handle_datetime_column(date_range)

    assert parsed_dt is not None
    assert dtype == "datetime"
    assert parsed_dt.isna().sum() == 3
    for i in range(len(date_range)):
        if i >= 0 and i < 3:
            assert pd.isnull(parsed_dt[i])
        else:
            assert parsed_dt[i] == pd.Timestamp(date_range[i])


def test_handle_datetime_column_nan_values_no_threshold():
    date_range = pd.Series(["{}.{}.20{:0>2}".format(i, i, i) for i in range(1, 10)])
    date_range[0:7] = ["abc" for _ in range(7)]
    parsed_dt, dtype = handle_datetime_column(date_range)

    assert parsed_dt is not None
    assert dtype == "others"
    assert (parsed_dt == date_range).all()
