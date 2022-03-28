import pytest

from actableai.tasks.data_imputation import (
    AAIDataImputationTask,
    construct_rules,
)
import pandas as pd
import numpy as np

np.random.seed(1)


@pytest.fixture(scope="function")
def data_imputation_task():
    yield AAIDataImputationTask(use_ray=False)


@pytest.fixture(scope="function")
def date_range():
    yield pd.date_range("2015-02-24", periods=20, freq="T")


class TestDataImputation:
    def test_impute_null(self, data_imputation_task):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
            }
        )

        r = data_imputation_task.run(df, ("", ""), impute_nulls=True)

        assert r["status"] == "SUCCESS"

    def test_single_row_rule(self, data_imputation_task):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
            }
        )

        r = data_imputation_task.run(df, ("", "y=1"))

        assert r["status"] == "SUCCESS"

    def test_multi_row_rule(self, data_imputation_task):
        df = pd.DataFrame(
            {
                "a": ["a", "b", "c", "d", "a", "a", "b", "c", "d", "e"] * 2,
                "b": ["a", "b", "c", "d", "a", "a", "b", "c", "d", "e"] * 2,
                "c": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
                "d": [1, 2, 1, 2, 1, None, 3, 3, 1, 2] * 2,
            }
        )

        r = data_imputation_task.run(df, ("a=b->c<>d", ""))

        assert r["status"] == "SUCCESS"

    def test_mix_single_and_multi_row_rule(self, data_imputation_task):
        df = pd.DataFrame(
            {
                "a": ["a", "b", "c", "d", "a", "a", "b", "c", "d", "e"] * 2,
                "b": ["a", "b", "c", "d", "a", "a", "b", "c", "d", "e"] * 2,
                "c": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
                "d": [1, 2, 1, 2, 1, None, 3, 3, 1, 2] * 2,
            }
        )

        r = data_imputation_task.run(df, ("a=a OR a=b->c<>d", ""))

        assert r["status"] == "SUCCESS"

    def test_impute_timeseries(self, data_imputation_task, date_range):
        df = pd.DataFrame(
            {
                "Date": date_range,
                "x": ["a", "a", "a", "c", "b", "b", "c", "b", None, "b"] * 2,
            }
        )

        r = data_imputation_task.run(df)

        assert r["status"] == "SUCCESS"

    def test_data_contain_datetime_column(self, data_imputation_task, date_range):
        df = pd.DataFrame(
            {
                "a": ["a", "b", "c", "d", "a", "a", "b", "c", "d", "e"] * 2,
                "b": ["a", "b", "c", "d", "a", "a", "b", "c", "d", "e"] * 2,
                "c": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
                "d": date_range,
            }
        )

        r = data_imputation_task.run(df, ("", ""))

        assert r["status"] == "SUCCESS"

    def test_impute_boolean(self, data_imputation_task):
        df = pd.DataFrame({"x": [True, False, np.nan, True, None] * 2})

        r = data_imputation_task.run(df)

        assert r["status"] == "SUCCESS"

    def test_impute_datetime(self, data_imputation_task, date_range):
        df = pd.DataFrame(
            {
                "Date": date_range,
                "x": ["a", "a", "a", "c", "b", "b", "c", "b", None, "b"] * 2,
            }
        )
        drop_indices = np.random.randint(0, 20, 10)
        df.iloc[drop_indices, :] = None

        r = data_imputation_task.run(df)

        assert r["status"] == "SUCCESS"
        assert len(r["data"]["records"]) == len(date_range)
        for idx in drop_indices:
            assert r["data"]["records"][idx]["text"]["Date"] == str(date_range[idx])


@pytest.mark.parametrize(
    "raw, expect_rule_str",
    [
        (None, ("", "")),
        (
            [
                {
                    "title": "Arule",
                    "validations": [],
                    "misplaced": [],
                }
            ],
            ("", ""),
        ),
        (
            [
                {
                    "title": "Arule",
                    "validations": [
                        {
                            "when": [
                                {
                                    "column": {
                                        "label": "HospitalName",
                                        "value": "HospitalName",
                                    },
                                    "operator": {
                                        "label": "=",
                                        "value": "=",
                                    },
                                    "comparedColumn": {
                                        "label": "HospitalName",
                                        "value": "HospitalName",
                                    },
                                }
                            ],
                            "then": [],
                        }
                    ],
                    "misplaced": [],
                }
            ],
            (
                "",
                "",
            ),
        ),
        (
            [
                {
                    "title": "Arule",
                    "validations": [
                        {
                            "when": [
                                {
                                    "column": {
                                        "label": "HospitalName",
                                        "value": "HospitalName",
                                    },
                                    "operator": {
                                        "label": "=",
                                        "value": "=",
                                    },
                                    "comparedColumn": {
                                        "label": "HospitalName",
                                        "value": "HospitalName",
                                    },
                                }
                            ],
                            "then": [
                                {
                                    "column": {
                                        "label": "ProviderNumber",
                                        "value": "ProviderNumber",
                                    },
                                    "operator": {"label": "<>", "value": "<>"},
                                    "comparedColumn": {
                                        "label": "ProviderNumber",
                                        "value": "ProviderNumber",
                                    },
                                }
                            ],
                        }
                    ],
                    "misplaced": [],
                }
            ],
            (
                "HospitalName=HospitalName -> ProviderNumber<>ProviderNumber",
                "",
            ),
        ),
        (
            [
                {
                    "title": "Arule",
                    "validations": [
                        {
                            "when": [
                                {
                                    "column": {
                                        "label": "HospitalName",
                                        "value": "HospitalName",
                                    },
                                    "operator": {
                                        "label": "=",
                                        "value": "=",
                                    },
                                    "comparedColumn": {
                                        "label": "HospitalName",
                                        "value": "HospitalName",
                                    },
                                }
                            ],
                            "then": [
                                {
                                    "column": {
                                        "label": "ProviderNumber",
                                        "value": "ProviderNumber",
                                    },
                                    "operator": {"label": "<>", "value": "<>"},
                                    "comparedColumn": {
                                        "label": "ProviderNumber",
                                        "value": "ProviderNumber",
                                    },
                                },
                                {
                                    "column": {
                                        "label": "Address1",
                                        "value": "Address1",
                                    },
                                    "operator": {"label": "<>", "value": "<>"},
                                    "comparedColumn": {
                                        "label": "Address1",
                                        "value": "Address1",
                                    },
                                },
                            ],
                        },
                    ],
                    "misplaced": [],
                }
            ],
            (
                "HospitalName=HospitalName -> ProviderNumber<>ProviderNumber "
                "& Address1<>Address1",
                "",
            ),
        ),
        (
            [
                {
                    "title": "Arule",
                    "validations": [
                        {
                            "when": [
                                {
                                    "column": {
                                        "label": "HospitalName",
                                        "value": "HospitalName",
                                    },
                                    "operator": {"label": "=", "value": "="},
                                    "comparedColumn": {
                                        "label": "HospitalName",
                                        "value": "HospitalName",
                                    },
                                }
                            ],
                            "then": [
                                {
                                    "column": {
                                        "label": "ProviderNumber",
                                        "value": "ProviderNumber",
                                    },
                                    "operator": {"label": "<>", "value": "<>"},
                                    "comparedColumn": {
                                        "label": "ProviderNumber",
                                        "value": "ProviderNumber",
                                    },
                                }
                            ],
                        },
                        {
                            "when": [
                                {
                                    "column": {
                                        "label": "HospitalName",
                                        "value": "HospitalName",
                                    },
                                    "operator": {"label": "=", "value": "="},
                                    "comparedColumn": {
                                        "label": "HospitalName",
                                        "value": "HospitalName",
                                    },
                                },
                                {
                                    "column": {
                                        "label": "ProviderNumber",
                                        "value": "ProviderNumber",
                                    },
                                    "operator": {"label": "<>", "value": "<>"},
                                    "comparedColumn": {
                                        "label": "ProviderNumber",
                                        "value": "ProviderNumber",
                                    },
                                },
                            ],
                            "then": [
                                {
                                    "column": {
                                        "label": "Address1",
                                        "value": "Address1",
                                    },
                                    "operator": {"label": "<>", "value": "<>"},
                                    "comparedColumn": {
                                        "label": "Address1",
                                        "value": "Address1",
                                    },
                                }
                            ],
                        },
                    ],
                    "misplaced": [],
                }
            ],
            (
                "HospitalName=HospitalName -> ProviderNumber<>ProviderNumber "
                "OR HospitalName=HospitalName & ProviderNumber<>ProviderNumber -> Address1<>Address1",
                "",
            ),
        ),
        (
            [
                {
                    "title": "Arule",
                    "validations": [],
                    "misplaced": [
                        {
                            "column": {
                                "label": "ProviderNumber",
                                "value": "ProviderNumber",
                            },
                            "operator": {"label": "<>", "value": "<>"},
                            "value": "any value",
                            "isRegex": False,
                        }
                    ],
                }
            ],
            (
                "",
                "ProviderNumber<>any value",
            ),
        ),
        (
            [
                {
                    "title": "Arule",
                    "validations": [],
                    "misplaced": [
                        {
                            "column": {
                                "label": "ProviderNumber",
                                "value": "ProviderNumber",
                            },
                            "operator": {"label": "<>", "value": "<>"},
                            "value": "any value",
                            "isRegex": True,
                        }
                    ],
                }
            ],
            (
                "",
                "ProviderNumber<>s/any value/g",
            ),
        ),
        (
            [
                {
                    "title": "Arule",
                    "validations": [],
                    "misplaced": [
                        {
                            "column": {
                                "label": "ProviderNumber",
                                "value": "ProviderNumber",
                            },
                            "operator": {"label": "<>", "value": "<>"},
                            "value": "any value",
                            "isRegex": True,
                        },
                        {
                            "column": {"label": "Address1", "value": "Address1"},
                            "operator": {"label": "=", "value": "="},
                            "value": "any address",
                            "isRegex": True,
                        },
                    ],
                }
            ],
            (
                "",
                "ProviderNumber<>s/any value/g OR Address1=s/any address/g",
            ),
        ),
        (
            [
                {
                    "title": "Arule",
                    "validations": [
                        {
                            "when": [
                                {
                                    "column": {
                                        "label": "HospitalName",
                                        "value": "HospitalName",
                                    },
                                    "operator": {"label": "=", "value": "="},
                                    "comparedColumn": {
                                        "label": "HospitalName",
                                        "value": "HospitalName",
                                    },
                                }
                            ],
                            "then": [
                                {
                                    "column": {
                                        "label": "ProviderNumber",
                                        "value": "ProviderNumber",
                                    },
                                    "operator": {"label": "<>", "value": "<>"},
                                    "comparedColumn": {
                                        "label": "ProviderNumber",
                                        "value": "ProviderNumber",
                                    },
                                }
                            ],
                        }
                    ],
                    "misplaced": [
                        {
                            "column": {
                                "label": "ProviderNumber",
                                "value": "ProviderNumber",
                            },
                            "operator": {"label": "<>", "value": "<>"},
                            "value": "any value",
                            "isRegex": True,
                        },
                    ],
                }
            ],
            (
                "HospitalName=HospitalName -> ProviderNumber<>ProviderNumber",
                "ProviderNumber<>s/any value/g",
            ),
        ),
    ],
)
def test_construct_rules(raw, expect_rule_str):
    rule = construct_rules(raw)
    assert rule == expect_rule_str
