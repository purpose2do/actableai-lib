import numpy as np
import pandas as pd


def empty_string_to_nan(
    df: pd.DataFrame,
    task_model,
    intervention_col: str,
    new_intervention_col: str,
    expected_target: str,
) -> pd.DataFrame:
    """Replace the empty string to nan in interventional model, also checks if the
        column exist in the DataFrame for new_intervention_col


    Args:
        df: DataFrame with empty strings values
        task_model: Task_model containing the discrete_treatment information
        intervention_col: Column name of the current treatment
        new_intervention_col: Column name of the new treatment

    Returns:
        pd.DataFrame: DataFrame containing the result with the intervention columns
            empty strings set to nan values and casted as float if treatment is not
            discrete.
    """
    df[intervention_col] = df[intervention_col].replace("", np.nan)
    if new_intervention_col in df:
        df[new_intervention_col] = df[new_intervention_col].replace("", np.nan)
    if expected_target in df:
        df[expected_target] = df[expected_target].replace("", np.nan).astype(float)
    if not task_model.intervention_model.causal_model.discrete_treatment:
        df[intervention_col] = df[intervention_col].astype(float)
        if new_intervention_col in df:
            df[new_intervention_col] = df[new_intervention_col].astype(float)
    return df
