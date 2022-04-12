from typing import List, Tuple
import pandas as pd

def leaderboard_cross_val(cross_val_leaderboard:List[pd.DataFrame]) -> pd.DataFrame:
    conc_leaderboard = pd.concat(cross_val_leaderboard)
    avg = conc_leaderboard.groupby('model').mean().reset_index()
    std = conc_leaderboard.groupby('model').std().reset_index()
    std = std.add_suffix('_std')
    return avg.join(std)
