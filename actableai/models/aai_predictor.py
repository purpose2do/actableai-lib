from typing import Optional
from dataclasses import dataclass
from autogluon.tabular import TabularPredictor
from econml.dml import DML


@dataclass
class AAIPredictor:
    version: int
    predictor: TabularPredictor
    causal_model: Optional[DML] = None
    intervened_column: Optional[str] = None
