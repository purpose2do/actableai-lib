from actableai.utils.categorical_numerical_convert import (
    convert_categorical_to_num,
    inverse_convert_categorical_to_num,
)
import pandas as pd


def test_categorical_to_num():
    """
    Check if conversion of categorical features to enumerated, and conversion back to numerical yields the same dataframe as the original.
    """

    df = pd.read_csv(
        "https://raw.githubusercontent.com/Actable-AI/public-datasets/master/apartments.csv"
    )

    df_conv, df_conv_uniques = convert_categorical_to_num(df)
    df_conv_inverse = inverse_convert_categorical_to_num(
        df_conv, df_conv_uniques, feat_name=None
    )

    assert df_conv_inverse.equals(df)
