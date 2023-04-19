from typing import Dict, Type

from .ag_automm import AGAUTOMMParams
from .base import Model, BaseParams
from .cat import CATParams
from .fastainn import FastAINNParams
from .fasttext import FASTTEXTParams
from .gbm import GBMParams
from .knn import KNNParams
from .lr import LRParams
from .nn_mxnet import NNMXNetParams
from .nn_torch import NNTorchParams
from .rf import RFParams
from .tabpfn import TabPFNParams
from .xgboost_linear import XGBoostLinearParams
from .xgboost_tree import XGBoostTreeParams
from .xt import XTParams

model_params_dict: Dict[Model, Type[BaseParams]] = {
    Model.gbm: GBMParams,
    Model.cat: CATParams,
    Model.xgb_tree: XGBoostTreeParams,
    Model.xgb_linear: XGBoostLinearParams,
    Model.rf: RFParams,
    Model.xt: XTParams,
    Model.knn: KNNParams,
    Model.lr: LRParams,
    Model.nn_torch: NNTorchParams,
    Model.nn_mxnet: NNMXNetParams,
    Model.nn_fastainn: FastAINNParams,
    Model.ag_automm: AGAUTOMMParams,
    Model.fasttext: FASTTEXTParams,
    Model.tabpfn: TabPFNParams,
}
