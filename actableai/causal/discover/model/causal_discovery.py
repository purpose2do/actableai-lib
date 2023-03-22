from pydantic import BaseModel

from actableai.causal.discover.algorithms.payloads import CausalVariableNature


class NormalizedColumnMetadata(BaseModel):
    upper: float
    lower: float
    mean: float
    std: float


class DatasetStatistics(BaseModel):
    number_of_dropped_rows: int
    number_of_rows: int


_causal_var_nature_to_causica_var_type = {
    "Discrete": "continuous",  # TODO: make categorical
    "Continuous": "continuous",
    "Categorical Ordinal": "continuous",  # TODO: make categorical
    "Categorical Nominal": "continuous",  # TODO: make categorical
    "Binary": "binary",
    "Excluded": "continuous",
}


def map_to_causica_var_type(nature: CausalVariableNature):
    return _causal_var_nature_to_causica_var_type.get(nature, "continuous")
