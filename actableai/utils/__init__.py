from typing import Optional
import numpy as np
import os
import pandas as pd
import uuid
from copy import deepcopy
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics._ranking import _binary_clf_curve


def fill_na(df, fillna_dict=None, fill_median=True):
    import numpy as np

    if fillna_dict is None:
        fillna_dict = {object: "", int: np.nan, float: np.nan}

    cat_cols = df.select_dtypes(exclude=["number"]).columns
    ordinal_cols = df.select_dtypes(include=["number"]).columns
    df[cat_cols] = df[cat_cols].fillna(fillna_dict[object])
    if fill_median:
        for col in ordinal_cols:
            df[col] = df[col].fillna(df[col].median())
    else:
        df[ordinal_cols] = df[ordinal_cols].fillna(fillna_dict[int])

    return df


def handle_datetime_features(df):
    datetime_cols = df.select_dtypes(include=["datetime"]).columns
    for col in datetime_cols:
        df[col] = pd.to_numeric(df[col])
    return df


def handle_boolean_features(df):
    from pandas.api.types import infer_dtype

    for col in df.columns:
        dtype = infer_dtype(df[col])
        if dtype == "boolean":
            df[col][pd.isnull(df[col])] = ""
            df[col] = df[col].astype(str)
            df[col][df[col] == ""] = np.NaN
    return df


def preprocess_dataset(df):
    df = handle_datetime_features(df)
    df = handle_boolean_features(df)
    df = fill_na(df)
    return df


def get_type_special(X: pd.Series) -> str:
    from autogluon.common.features.infer_types import (
        check_if_datetime_as_object_feature,
        check_if_nlp_feature,
    )
    from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, infer_dtype
    import numpy as np

    type_special = "unknown"
    if len(X) > 0:
        if "mixed" in infer_dtype(X):
            type_special = "mixed"
        elif infer_dtype(X) == "boolean":
            type_special = "boolean"
        elif is_datetime64_any_dtype(X):
            type_special = "datetime"
        elif check_if_datetime_as_object_feature(X):
            type_special = "datetime"
        elif check_if_nlp_feature(X):
            type_special = "text"
        elif X.dtype == np.dtype("O"):
            type_special = "category"
        elif check_if_integer_feature(X):
            type_special = "integer"
        elif is_numeric_dtype(X):
            type_special = "numeric"
    elif len(X) == 0:
        type_special = "empty"
    return type_special


def get_type_special_no_ag(X: pd.Series) -> str:
    """
    From autogluon library
    TODO improve
    """
    from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, infer_dtype
    import numpy as np
    import logging

    def get_type_family_raw(dtype) -> str:
        """From dtype, gets the dtype family."""
        try:
            if dtype.name == "category":
                return "category"
            if "datetime" in dtype.name:
                return "datetime"
            elif np.issubdtype(dtype, np.integer):
                return "int"
            elif np.issubdtype(dtype, np.floating):
                return "float"
        except Exception as err:
            logging.error(
                f"Warning: dtype {dtype} is not recognized as a valid dtype by numpy! AutoGluon may incorrectly handle this feature..."
            )
            logging.error(err)

        if dtype.name in ["bool", "bool_"]:
            return "bool"
        elif dtype.name in ["str", "string", "object"]:
            return "object"
        else:
            return dtype.name

    def check_if_datetime_as_object_feature(X: pd.Series) -> bool:
        type_family = get_type_family_raw(X.dtype)
        if X.isnull().all():
            return False
        if type_family != "object":
            return False
        try:
            X.apply(pd.to_numeric)
        except Exception:
            try:
                X.apply(pd.to_datetime)
                return True
            except Exception:
                return False
        else:
            return False

    def check_if_nlp_feature(X: pd.Series) -> bool:
        if X.isna().all():
            return False
        type_family = get_type_family_raw(X.dtype)
        if type_family != "object":
            return False
        X_unique = X.unique()
        num_unique = len(X_unique)
        num_rows = len(X)
        unique_ratio = num_unique / num_rows
        if unique_ratio <= 0.01:
            return False
        try:
            avg_words = pd.Series(X_unique).str.split().str.len().mean()
        except AttributeError:
            return False
        if avg_words < 3:
            return False

        return True

    def check_if_integer_feature(X: pd.Series):
        import numpy as np

        clean_X = X.dropna()
        return np.array_equal(clean_X.values, clean_X.values.astype(int))

    type_special = "unknown"
    if len(X) > 0:
        if "mixed" in infer_dtype(X):
            type_special = "mixed"
        elif infer_dtype(X) == "boolean":
            type_special = "boolean"
        elif is_datetime64_any_dtype(X):
            type_special = "datetime"
        elif check_if_datetime_as_object_feature(X):
            type_special = "datetime"
        elif check_if_nlp_feature(X):
            type_special = "text"
        elif X.dtype == np.dtype("O"):
            type_special = "category"
        elif check_if_integer_feature(X):
            type_special = "integer"
        elif is_numeric_dtype(X):
            type_special = "numeric"
    elif len(X) == 0:
        type_special = "empty"
    return type_special


def check_if_integer_feature(X: pd.Series):
    import numpy as np

    clean_X = X.dropna()
    return np.array_equal(clean_X.values, clean_X.values.astype(int))


def memory_efficient_hyperparameters(
    ag_automm_enabled: bool = False, tabpfn_enabled: bool = False
):
    from autogluon.tabular.configs.hyperparameter_configs import (
        hyperparameter_config_dict,
    )
    from autogluon.text.text_prediction.presets import list_text_presets
    from actableai.classification.models import TabPFNModel

    # Returns autogluon tabular predictor's hyperparameters without the heavy-memory models

    hyperparameters = deepcopy(hyperparameter_config_dict["default"])
    if "NN_TORCH" in hyperparameters:
        del hyperparameters["NN_TORCH"]

    # Text models
    hyperparameters["FASTTEXT"] = {}

    simple_presets = list_text_presets(verbose=True)
    # Change the batch size if we encounter memory issues
    if ag_automm_enabled:
        hyperparameters["AG_AUTOMM"] = simple_presets["multilingual"]  # type: ignore
    # hyperparameters["AG_AUTOMM"]["env.per_gpu_batch_size"] = 4
    if tabpfn_enabled:
        hyperparameters[TabPFNModel] = {}

    return hyperparameters


def fast_categorical_hyperparameters():
    from autogluon.tabular.configs.hyperparameter_configs import (
        hyperparameter_config_dict,
    )

    # Returns autogluon tabular predictor's hyperparameters for fast training with cat vars

    hyperparameters = deepcopy(hyperparameter_config_dict["default"])
    if "NN" in hyperparameters:
        del hyperparameters["NN"]
    if "GBM" in hyperparameters:
        del hyperparameters["GBM"]
    if "XGB" in hyperparameters:
        del hyperparameters["XGB"]

    return hyperparameters


def debiasing_hyperparameters():
    return {"LR": {"proc.skew_threshold": np.inf}}


def quantile_regression_hyperparameters():
    return {
        "FASTAI": {},
        "NN_TORCH": {},
    }


def explanation_hyperparameters():
    from autogluon.tabular.configs.hyperparameter_configs import (
        hyperparameter_config_dict,
    )

    hyperparameters = deepcopy(hyperparameter_config_dict["default"])

    compatible_models = {"GBM", "CAT", "XGB", "RF", "XT"}
    for model_name in hyperparameter_config_dict["default"].keys():
        if model_name not in compatible_models:
            del hyperparameters[model_name]

    return hyperparameters


def debiasing_feature_generator_args():
    return {
        "enable_numeric_features": True,
        "enable_categorical_features": True,
        "enable_datetime_features": True,
        "enable_text_special_features": False,
        "enable_text_ngram_features": False,
        "enable_raw_text_features": False,
        "enable_vision_features": True,
        # "pre_drop_useless": False,
        # "post_generators": [],
    }


def random_directory(path=""):
    """
    Create random directory,
    """
    uid = uuid.uuid4()
    directory = os.path.join(path, str(uid))

    return directory


def is_fitted(transformer):
    try:
        check_is_fitted(transformer)
    except NotFittedError:
        return False
    return True


def is_text_column(X, text_ratio=0.1):
    X_unique = X.unique()
    num_unique = len(X_unique)
    num_rows = len(X)
    unique_ratio = num_unique / num_rows
    return unique_ratio > text_ratio


def custom_precision_recall_curve(
    y_true, probas_pred, *, pos_label=None, sample_weight=None
):
    fps, tps, thresholds = _binary_clf_curve(
        y_true, probas_pred, pos_label=pos_label, sample_weight=sample_weight
    )

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]

    sl = slice(-1, None, -1)
    return precision[sl], recall[sl], thresholds[sl]
