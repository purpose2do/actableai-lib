from typing import Iterable, Tuple, Any

import numpy as np
import pandas as pd
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

from actableai.timeseries.transform.base import ArrayTransformation


class PowerTransformation(ArrayTransformation):
    """Power transformation.

    This transformation will have for effect to make the data more Gaussian-like.
    see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html
    """

    def __init__(self):
        """PowerTransformation constructor."""
        super().__init__()

        self._transformation_models = None

    @staticmethod
    def _train_transformation_model(data: DataEntry) -> Pipeline:
        """Train model representing the power transformation.

        Args:
            data: Data used for training.

        Returns:
            The trained model.
        """
        X = data[FieldName.TARGET]

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        model = make_pipeline(
            StandardScaler(with_std=False),
            PowerTransformer(standardize=True),
        )
        model.fit(X)

        return model

    def _setup_data(self, data_it: Iterable[DataEntry]):
        """Set up the transformation with data.

        Args:
            data_it: Data to set up the transformation with.
        """
        super()._setup_data(data_it)

        self._transformation_models = {
            group: self._train_transformation_model(data)
            for data, group in zip(data_it, self.group_list)
        }

    def transform_array(
        self, array: np.ndarray, start_date: pd.Period, group: Tuple[Any, ...]
    ) -> np.ndarray:
        """Transform an array.

        Args:
            array: Array to transform.
            start_date: Starting date of the array (in the time series context).
            group: Array's group.

        Returns:
            The transformed array.
        """
        return self._transformation_models[group].transform(array.T).T

    def revert_array(
        self, array: np.ndarray, start_date: pd.Period, group: Tuple[Any, ...]
    ) -> np.ndarray:
        """Revert a transformation on an array.

        Args:
            array: Array to revert.
            start_date: Starting date of the array (in the time series context).
            group: Array's group.

        Returns:
            The transformed array.
        """
        return self._transformation_models[group].inverse_transform(array.T).T
