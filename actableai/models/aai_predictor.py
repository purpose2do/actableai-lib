from typing import Dict, List, Optional
from autogluon.tabular import TabularPredictor
import pandas as pd

from actableai.intervention import custom_intervention_effect


class AAIModel:
    model_version = 1

    def __init__(self, version: int = model_version) -> None:
        self.version = version


class AAITabularModel(AAIModel):
    def __init__(
        self,
        version: int,
        predictor: TabularPredictor,
    ) -> None:
        super().__init__(version)
        self.predictor = predictor


class AAIInterventionalModel(AAIModel):
    def __init__(
        self,
        version: int,
        causal_model,
        outcome_transformer,
        discrete_treatment,
        common_causes,
        intervened_column,
    ) -> None:
        super().__init__(version)
        self.common_causes = common_causes
        self.causal_model = causal_model
        self.outcome_transformer = outcome_transformer
        self.discrete_treatment = discrete_treatment
        self.intervened_column = intervened_column


class AAITabularModelInterventional(AAITabularModel):
    def __init__(
        self,
        version: int,
        predictor: TabularPredictor,
        intervention_model: AAIInterventionalModel,
    ) -> None:
        super().__init__(
            version,
            predictor,
        )
        self.intervention_model = intervention_model

    def intervention_effect(self, df: pd.DataFrame, pred: Dict) -> pd.DataFrame:
        return custom_intervention_effect(
            df=df,
            pred=pred,
            predictor=self.predictor,
            aai_interventional_model=self.intervention_model,
        )
