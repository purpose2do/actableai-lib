import pandas as pd
from typing import List, Tuple


def leaderboard_cross_val(cross_val_leaderboard: List[pd.DataFrame]) -> pd.DataFrame:
    """Creates a leaderboard from a list of cross validation results.

    Args:
        cross_val_leaderboard: List of cross validation results.

    Returns:
        pd.DataFrame: Leaderboard.
    """
    conc_leaderboard = pd.concat(cross_val_leaderboard)
    avg = conc_leaderboard.groupby("model").mean().reset_index()
    std = conc_leaderboard.groupby("model").std().reset_index()
    std = std.add_suffix("_std")
    return avg.join(std)


def split_validation_by_datetime(
    df_train: pd.DataFrame,
    datetime_column: str,
    validation_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sorted_df = df_train.sort_values(by=datetime_column, ascending=True)
    split_datetime_index = int((1 - validation_ratio) * len(sorted_df))
    df_train = sorted_df.iloc[:split_datetime_index].sample(frac=1)
    df_val = sorted_df.iloc[split_datetime_index:]
    return df_train, df_val
