from typing import Dict, List, Optional
from autogluon.tabular import TabularPredictor
from econml.dml import DML
import pandas as pd


class AAIModel:
    model_version = 1

    def __init__(self, version: int = model_version) -> None:
        self.version = version


class AAITabularModel(AAIModel):
    def __init__(
        self,
        version: int,
        predictor: TabularPredictor,
        causal_model: Optional[DML],
        intervened_column: Optional[str],
        common_causes: Optional[List[str]],
    ) -> None:
        super().__init__(version)
        self.predictor = predictor
        self.causal_model = causal_model
        self.intervened_column = intervened_column
        self.common_causes = common_causes

    def intervention_effect(self, df: pd.DataFrame, pred: Dict) -> pd.DataFrame:
        if self.causal_model and self.intervened_column and self.common_causes:
            CME = self.causal_model.const_marginal_effect(
                df[self.common_causes]
            ).squeeze()
            ctr, cta = (
                df[self.intervened_column],
                pred["prediction"],
            )
            # New Outcome
            new_out = [None for _ in range(len(df))]
            if f"intervened_{self.intervened_column}" in df:
                ntr = df[f"intervened_{self.intervened_column}"]
                new_out = (ntr - ctr) * CME + cta
            # New Intervention
            new_inter = [None for _ in range(len(df))]
            if f"expected_{self.predictor.label}" in df:
                nta = df[f"expected_{self.predictor.label}"]
                new_inter = ((nta - cta) / CME) + ctr
            return pd.DataFrame({"new_outcome": new_out, "new_intervention": new_inter})
        else:
            raise Exception()
