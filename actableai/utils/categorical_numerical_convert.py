def get_categorical_columns(df):
    # Check which columns are not integers or floats, and assume they are categorical features to be converted
    return df.dtypes[df.dtypes != "int64"][df.dtypes != "float64"]

def convert_categorical_to_num(df_orig):
    """
    Convert categorical features in a dataframe to numerical values.

    Parameters:
    df_orig (pandas DataFrame): The dataframe containing the categorical features.

    Returns:
    df (pandas DataFrame): The modified DataFrame with categorical features converted to numerical values.
    uniques_all (dict): A dictionary containing the unique values for each converted column (the categorical features).
    """

    import pandas as pd
    import copy

    df = copy.deepcopy(df_orig)

    # Get categorical columns/features
    cols = get_categorical_columns(df)

    # Convert features to enumerated types
    uniques_all = dict()
    for c in cols.keys():
        df[c], uniques_all[c] = pd.factorize(df[c])

    return df, uniques_all


def inverse_convert_categorical_to_num(df_new, uniques, feat_name=None):
    """
    Convert numerical values back to their original categorical values.

    This function takes in a DataFrame and a dictionary of unique values for each column, and converts the numerical values in the DataFrame back to their original categorical values. It can be used to reverse the effect of the convert_categorical_to_num function.

    Parameters:
    df_new (pandas DataFrame): The DataFrame containing the numerical values to be converted back to categorical.
    uniques (dict): A dictionary containing the unique values for each column.
    feat_name (str, optional): The name of a specific feature to be converted. If None, all features in the DataFrame will be converted.

    Returns:
    df (pandas DataFrame): The modified DataFrame with numerical values converted back to categorical values.

    Raises:
    ValueError: If the feature name (column) is not in the DataFrame.
    """

    import copy
    import numpy as np

    df = copy.deepcopy(df_new)

    if feat_name == None:  # Iterate over ALL features
        for c in uniques.keys():
            df[c] = uniques[c].take(df_new[c])
            df.loc[
                df_new[c] == -1, c
            ] = (
                np.nan
            )  # NaN values are indicated with the -1 sentinel; replace them with NaN

    else:  # Only one feature
        c = feat_name
        if c in df_new.columns:
            df[c] = uniques[c].take(df_new[c])
            df[
                df_new[c] == -1
            ] = (
                np.nan
            )  # NaN values are indicated with the -1 sentinel; replace them with NaN

        else:
            raise ValueError("Feature (column) not in dataframe!")

    return df
