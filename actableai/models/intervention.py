import numpy as np


def empty_string_to_nan(df, task_model, intervention_col, new_intervention_col):
    df[intervention_col] = df[intervention_col].replace("", np.nan)
    if new_intervention_col in df:
        df[new_intervention_col] = df[new_intervention_col].replace("", np.nan)
    if not task_model.intervention_model.causal_model.discrete_treatment:
        df[intervention_col] = df[intervention_col].astype(float)
        if new_intervention_col in df:
            df[new_intervention_col] = df[new_intervention_col].astype(float)
    return df
