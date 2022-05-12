import pandas as pd
from typing import List


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
