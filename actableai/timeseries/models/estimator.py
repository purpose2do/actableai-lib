from gluonts.model.estimator import Estimator
from gluonts.model.predictor import Predictor

from actableai.timeseries.dataset import AAITimeSeriesDataset


class AAITimeSeriesEstimator:
    """Custom Wrapper around GluonTS Estimator."""

    def __init__(
        self,
        estimator: Estimator,
        keep_feat_static_real: bool = True,
        keep_feat_static_cat: bool = True,
        keep_feat_dynamic_real: bool = True,
        keep_feat_dynamic_cat: bool = True,
    ):
        """AAITimeSeriesEstimator Constructor.

        Args:
            estimator: Underlying GluonTS estimator.
            keep_feat_static_real: If False the real static features will be filtered
                out.
            keep_feat_static_cat: If False the categorical static features will be
                filtered out.
            keep_feat_dynamic_real: If False the real dynamic features will be filtered
                out.
            keep_feat_dynamic_cat: If False the categorical dynamic features will be
                filtered out.
        """
        self.estimator = estimator

        self.keep_feat_static_real = keep_feat_static_real
        self.keep_feat_static_cat = keep_feat_static_cat
        self.keep_feat_dynamic_real = keep_feat_dynamic_real
        self.keep_feat_dynamic_cat = keep_feat_dynamic_cat

    def train(
        self,
        training_dataset: AAITimeSeriesDataset,
        validation_dataset: AAITimeSeriesDataset = None,
    ) -> Predictor:
        """Train estimator.

        Args:
            training_dataset: Training data.
            validation_dataset: Validation data, used for tuning.

        Returns:
            GluonTS trained predictor.
        """
        training_dataset = training_dataset.clean_features(
            self.keep_feat_static_real,
            self.keep_feat_static_cat,
            self.keep_feat_dynamic_real,
            self.keep_feat_dynamic_cat,
        )
        training_dataset.training = True

        if validation_dataset is not None:
            validation_dataset = validation_dataset.clean_features(
                self.keep_feat_static_real,
                self.keep_feat_static_cat,
                self.keep_feat_dynamic_real,
                self.keep_feat_dynamic_cat,
            )
            validation_dataset.training = True

        return self.estimator.train(training_dataset, validation_dataset)
