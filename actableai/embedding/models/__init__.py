from typing import Dict, Type

from .base import Model, BaseEmbeddingModel
from .linear_discriminant_analysis import LinearDiscriminantAnalysis
from .tsne import TSNE
from .umap import UMAP

model_dict: Dict[Model, Type[BaseEmbeddingModel]] = {
    Model.linear_discriminant_analysis: LinearDiscriminantAnalysis,
    Model.tsne: TSNE,
    Model.umap: UMAP,
}

model_parameters_dict = {
    model_name: model.get_parameters() for model_name, model in model_dict.items()
}
