from typing import Dict, Type

from .base import Model, BaseParams
from .constant_value import ConstantValueParams, MultivariateConstantValueParams
from .deep_ar import DeepARParams
from .deep_var import DeepVARParams
from .feed_forward import FeedForwardParams
from .gp_var import GPVarParams
from .n_beats import NBEATSParams
from .prophet import ProphetParams
from .r_forecast import RForecastParams
from .tree_predictor import TreePredictorParams

model_params_dict: Dict[Model, Type[BaseParams]] = {
    Model.constant_value: ConstantValueParams,
    Model.multivariate_constant_value: MultivariateConstantValueParams,
    Model.deep_ar: DeepARParams,
    Model.deep_var: DeepVARParams,
    Model.feed_forward: FeedForwardParams,
    Model.gp_var: GPVarParams,
    Model.n_beats: NBEATSParams,
    Model.prophet: ProphetParams,
    Model.r_forecast: RForecastParams,
    Model.tree_predictor: TreePredictorParams,
}

model_hyperparameters_dict = {
    model: params.get_hyperparameters() for model, params in model_params_dict.items()
}
