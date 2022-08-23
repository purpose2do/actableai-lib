from typing import Optional
from autogluon.tabular import TabularPredictor
from econml.dml import DML


class AAIPredictor:
    def __init__(
        self,
        version: int,
        predictor: TabularPredictor,
        causal_model: Optional[DML] = None,
    ):
        self.version = version
        self.predictor = predictor
        self.causal_model = causal_model
