from typing import Dict, Type

from .affinity_propagation import AffinityPropagation
from .agglomerative_clustering import AgglomerativeClustering
from .base import Model, BaseClusteringModel
from .dbscan import DBSCAN
from .dec import DEC
from .kmeans import KMeans
from .spectral_clustering import SpectralClustering

model_dict: Dict[Model, Type[BaseClusteringModel]] = {
    Model.affinity_propagation: AffinityPropagation,
    Model.agglomerative_clustering: AgglomerativeClustering,
    Model.dbscan: DBSCAN,
    Model.dec: DEC,
    Model.kmeans: KMeans,
    Model.spectral_clustering: SpectralClustering,
}

model_parameters_dict = {
    model_name: model.get_parameters() for model_name, model in model_dict.items()
}
