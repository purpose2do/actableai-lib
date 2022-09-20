from typing import Dict, List, Optional
from autogluon.tabular import TabularPredictor
from econml.dml import DML
import pandas as pd
import numpy as np


def custom_intervention_effect(
    df: pd.DataFrame,
    pred: Dict,
    causal_model: DML,
    intervened_column: str,
    common_causes: Optional[List[str]],
    predictor: TabularPredictor,
    discrete_treatment: Optional[bool],
) -> pd.DataFrame:
    cta = pred["prediction"][predictor.label].replace("", np.nan).astype(float)
    new_inter = [None for _ in range(len(df))]
    new_out = [None for _ in range(len(df))]
    if discrete_treatment:
        custom_effect = []
        for index_lab in range(len(df)):
            curr_discrete_effect = np.nan
            if isinstance(df[f"intervened_{intervened_column}"][index_lab], str):
                curr_discrete_effect = causal_model.effect(
                    df[common_causes][index_lab : index_lab + 1]
                    if common_causes is not None and len(common_causes) != 0
                    else None,
                    T0=df[[intervened_column]][index_lab : index_lab + 1],  # type: ignore
                    T1=df[[f"intervened_{intervened_column}"]][
                        index_lab : index_lab + 1
                    ],  # type: ignore
                )
                curr_discrete_effect = curr_discrete_effect.squeeze()  # type: ignore
            custom_effect.append(curr_discrete_effect)
        custom_effect = np.array(custom_effect).squeeze()
        new_out = [None for _ in range(len(df))]
        if f"intervened_{intervened_column}" in df:
            new_out = custom_effect + cta
        return pd.DataFrame(
            {
                f"expected_{predictor.label}": new_out,
                f"intervened_{intervened_column}": new_inter,
            }
        )
    ctr = df[intervened_column].replace("", np.nan).astype(float)
    custom_effect = []
    for index_lab in range(len(df)):
        curr_CME = causal_model.const_marginal_effect(
            df[common_causes][index_lab : index_lab + 1]
            if common_causes is not None and len(common_causes) != 0
            else None
        )
        curr_CME = curr_CME.squeeze()
        custom_effect.append(curr_CME)
    CME = np.array(custom_effect).squeeze()
    # New Outcome
    if f"intervened_{intervened_column}" in df:
        ntr = df[f"intervened_{intervened_column}"].replace("", np.nan).astype(float)
        new_out = (ntr - ctr) * CME + cta
    # New Intervention
    if f"expected_{predictor.label}" in df:
        nta = df[f"expected_{predictor.label}"].replace("", np.nan).astype(float)
        new_inter = ((nta - cta) / CME) + ctr
    return pd.DataFrame(
        {
            f"expected_{predictor.label}": new_out,
            f"intervened_{intervened_column}": new_inter,
        }
    )
