from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

from pydantic import BaseModel


class Constraints(BaseModel):
    causes: List[str]
    effects: List[str]
    forbiddenRelationships: List[Tuple[str, str]]
    potentialRelationships: Optional[List[Tuple[str, str]]] = None


class CausalVariableNature(str, Enum):
    Discrete = "Discrete"
    Continuous = "Continuous"
    CategoricalOrdinal = "Categorical Ordinal"
    CategoricalNominal = "Categorical Nominal"
    Binary = "Binary"
    Excluded = "Excluded"


class CausalVariable(BaseModel):
    name: str
    nature: Optional[CausalVariableNature] = None


class Dataset(BaseModel):
    data: Dict[str, List[Any]]


class NormalizationOptions(BaseModel):
    with_mean: bool = True
    with_std: bool = True


class CausalDiscoveryPayload(BaseModel):
    dataset: Dataset
    normalization: NormalizationOptions = NormalizationOptions()
    constraints: Constraints
    causal_variables: List[CausalVariable]

    class Config:
        arbitrary_types_allowed = True


class ATEDetails(BaseModel):
    reference: Optional[Union[float, str]] = None
    intervention: Optional[Union[float, str]] = None
    nature: Optional[CausalVariableNature] = None


# TODO: UPDATE DEFAULTS
class DeciModelOptions(BaseModel):
    base_distribution_type: Literal["gaussian", "spline"] = "spline"
    spline_bins: int = 8
    imputation: bool = False
    lambda_dag: float = 100.0
    lambda_sparse: float = 5.0
    tau_gumbel: float = 1.0
    var_dist_A_mode: Literal["simple", "enco", "true", "three"] = "three"
    imputer_layer_sizes: Optional[List[int]] = None
    mode_adjacency: Literal["upper", "lower", "learn"] = "learn"
    norm_layers: bool = True
    res_connection: bool = True
    encoder_layer_sizes: Optional[List[int]] = [32, 32]
    decoder_layer_sizes: Optional[List[int]] = [32, 32]
    cate_rff_n_features: int = 3000
    cate_rff_lengthscale: Union[int, float, List[float], Tuple[float, float]] = 1


# To speed up training you can try:
#  increasing learning_rate
#  increasing batch_size (reduces noise when using higher learning rate)
#  decreasing max_steps_auglag (go as low as you can and still get a DAG)
#  decreasing max_auglag_inner_epochs
# TODO: UPDATE DEFAULTS
class DeciTrainingOptions(BaseModel):
    learning_rate: float = 3e-2
    batch_size: int = 512
    standardize_data_mean: bool = False
    standardize_data_std: bool = False
    rho: float = 10.0
    safety_rho: float = 1e13
    alpha: float = 0.0
    safety_alpha: float = 1e13
    tol_dag: float = 1e-3
    progress_rate: float = 0.25
    max_steps_auglag: int = 20
    max_auglag_inner_epochs: int = 1000
    max_p_train_dropout: float = 0.25
    reconstruction_loss_factor: float = 1.0
    anneal_entropy: Literal["linear", "noanneal"] = "noanneal"


class DeciAteOptions(BaseModel):
    Ngraphs: Optional[int] = 1
    Nsamples_per_graph: Optional[int] = 5000
    most_likely_graph: Optional[int] = True
    ate_details_by_name: Dict[str, ATEDetails] = dict()


class DeciPayload(CausalDiscoveryPayload):
    model_options: DeciModelOptions = DeciModelOptions()
    training_options: DeciTrainingOptions = DeciTrainingOptions()
    ate_options: DeciAteOptions = DeciAteOptions()
    model_save_dir: str = None


class DirectLiNGAMPayload(CausalDiscoveryPayload):
    pass


class NotearsPayload(CausalDiscoveryPayload):
    max_iter: int = 100


class PCPayload(CausalDiscoveryPayload):
    variant: Literal["original", "stable"] = "original"
    alpha: float = 0.05
    ci_test: Literal["gauss", "g2", "chi2"] = "gauss"
