import pandas as pd
import numpy as np

from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

from actableai.utils import get_type_special


def get_categorical_columns(df):
    # Check which columns are not integers or floats, and assume they are categorical features to be converted
    cols_list = []
    for column in df.columns:
        col_type = get_type_special(df[column])
        if col_type == "category":
            cols_list.append(column)

    return cols_list


def convert_categorical_to_num(df, inplace=False):
    """
    Convert categorical features in a dataframe to numerical values.

    Parameters:
    df (pandas DataFrame): The dataframe containing the categorical features.
    inplace (bool, optional): Whether to perform modifications to df in-place

    Returns:
    df (pandas DataFrame): The modified DataFrame with categorical features converted to numerical values.
    dict_label_encoders (dict): A dictionary containing the fitted LabelEncoder object for each converted column (the categorical features).
    """

    if not inplace:
        df = df.copy()

    # Get categorical columns/features
    cols = get_categorical_columns(df)

    # Fit encoder and convert columns to enumerated types
    dict_label_encoders = defaultdict(LabelEncoder)
    df[cols] = df[cols].apply(lambda x: dict_label_encoders[x.name].fit_transform(x))

    return df, dict_label_encoders


def inverse_convert_categorical_to_num(df_new, d, feat_name=None):
    """
    Convert numerical values back to their original categorical values.

    This function takes in a DataFrame and a dictionary of unique values for each column, and converts the numerical values in the DataFrame back to their original categorical values. It can be used to reverse the effect of the convert_categorical_to_num function.

    Parameters:
    df (pandas DataFrame): The DataFrame containing the numerical values to be converted back to categorical.
    d (dict): A dictionary containing the fitted LabelEncoder object for each column.
    feat_name (str, optional): The name of a specific feature to be converted. If None, all features in the DataFrame will be converted.

    Returns:
    df (pandas DataFrame): The modified DataFrame with numerical values converted back to categorical values.

    Raises:
    ValueError: If the feature name (column) is not in the DataFrame.
    """

    if feat_name == None:  # Iterate over ALL features
        cols = list(d.keys())

    else:  # Only one feature
        if feat_name in df_new.columns:
            cols = [feat_name]

        else:
            raise ValueError("Feature (column) not in dataframe!")

    df_new[cols] = df_new[cols].apply(lambda x: d[x.name].inverse_transform(x))

    return df_new
