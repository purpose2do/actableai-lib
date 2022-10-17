from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.features.generators import (
    AutoMLPipelineFeatureGenerator,
    AbstractFeatureGenerator,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from abc import ABC, abstractmethod

from actableai.utils.multilabel_predictor import MultilabelPredictor


class UnsupportedPredictorType(ValueError):
    """Raised when the predictor type is not supported."""

    pass


class UnsupportedProblemType(ValueError):
    """Raised when the problem type is not supported."""

    pass


class DataFrameTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, column_names: Optional[List[str]] = None) -> None:
        """DataFrame Transformer to transform lists and np.ndarray in DataFrames

        Args:
            column_names: Names of the columns for
                new DataFrame, if None the columns be RangeIndex(0, n).
                Number of column names must be the same as the number of columns.
                Defaults to None.
        """
        super().__init__()
        self.column_names = column_names

    def fit(self, X):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X.tolist())
            if self.column_names is not None and len(self.column_names) > 0:
                df.columns = self.column_names
            return df
        if isinstance(X, List) and len(np.array(X).shape) != 2:
            df = pd.DataFrame(X)
            if self.column_names is not None:
                df.columns = self.column_names
            return df
        raise TypeError(X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)


class LinearRegressionWrapper(LinearRegression):
    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return {
            "r2": r2_score(y, y_pred, sample_weight=sample_weight),
            "y": y,
            "y_pred": y_pred,
            "sample_weight": sample_weight,
        }


class SKLearnWrapper(ABC):
    def __init__(
        self,
        ag_predictor,
        x_w_columns: Optional[List] = None,
        hyperparameters: Optional[Union[List, Dict]] = None,
        presets: Optional[str] = "best_quality",
        ag_args_fit: Optional[Dict] = None,
        feature_generator: Optional[AbstractFeatureGenerator] = None,
        holdout_frac: Optional[float] = None,
    ):
        """Construct a sklearn wrapper object

        Args:
            ag_predictor: AutoGluon Tabular Predictor
            x_w_columns: Name of common_causes and
                effect modifiers (order matters). Defaults to None.
            hyperparameters: HyperParameter for
                TabularPredictor. Defaults to None.
                presets (Optional[str], optional): Presets for TabularPredictor
                Defaults to "best_quality".
            ag_args_fit: Args fit for Tabular
                Predictor. Defaults to None.

        Raises:
            UnsupportedPredictorType: Ensure that we only use TabularPredictor or MultilabelPredictor
        """
        self.ag_predictor = ag_predictor
        if hyperparameters is None:
            self.hyperparameters = "default"
        else:
            self.hyperparameters = hyperparameters
        self.presets = presets
        self._df_transformer = DataFrameTransformer(x_w_columns)
        self.ag_args_fit = ag_args_fit
        self.train_data = None
        self.feature_generator = feature_generator
        self.holdout_frac = holdout_frac

        if self.feature_generator is None:
            self.feature_generator = AutoMLPipelineFeatureGenerator()

    def fit(self, X, y, sample_weight=None):
        """Fit the model

        Args:
            X (features, n): Features to fit the model
            y (n): Groundtruth to fit the model
            sample_weight: Optional sample weight to use in case of classification

        Returns:
            self: The fitted SKLearnWrapper
        """
        train_data = self._df_transformer.fit_transform(X)
        train_data = self.add_output(train_data, y)
        train_data = TabularDataset(train_data)
        self.set_sample_weight(sample_weight=sample_weight)

        self.ag_predictor.fit(
            train_data=train_data,
            presets=self.presets,
            hyperparameters=self.hyperparameters,
            ag_args_fit=self.ag_args_fit or {},
            feature_generator=self.feature_generator,
            holdout_frac=self.holdout_frac,
        )
        pd.set_option("chained_assignment", "warn")
        self.train_data = train_data
        return self

    def predict(self, X):
        """Infer the result from the data

        Args:
            X: Data used to make the prediction

        Returns:
            np.ndarray: Prediction
        """
        test_data = self._df_transformer.fit_transform(X)
        test_data = TabularDataset(test_data)
        y_pred = self.ag_predictor.predict(test_data).values
        return y_pred

    def predict_proba(self, X):
        """Predict the probabilities of classes if classification problem else returns
            the result.

        Args:
            X (features, n): Data to predict each probability class

        Returns:
            np.ndarray (class_number, n): Probabilities for each class
        """
        test_data = self._df_transformer.fit_transform(X)
        y_pred_proba = self.ag_predictor.predict_proba(test_data)
        return y_pred_proba.values

    @abstractmethod
    def set_sample_weight(self, sample_weight=None):
        """Set samples weight for the model

        Args:
            sample_weight: Sample weights for each class
        """
        raise NotImplementedError()

    @abstractmethod
    def add_output(self, train_data, y):
        """Method to add the output to the train_data. Necessary for AutoGLuon

        Args:
            train_data: Training data
            y: Groundtruth target

        Returns:
            pd.DataFrame: Original DataFrame with target added
        """
        raise NotImplementedError()

    @abstractmethod
    def score(self, X, y, sample_weight=None) -> Dict[str, Any]:
        """Score for the model

        Args:
            X: Values on which the score is calculated
            y: Groundtruth values for the target
            sample_weight: Sample weight for each class

        Raises:
            UnsupportedProblemType: If the problem_type of the TabularPredictor is not
                right

        Returns:
            Dict: Score results
                - accuracy | r2: Score metric
                - y: Original data used for calculating the metric
                - y_pred: Predicted data uised for calculating the metric
                - sample_weight: Weights used for each class if usable
        """
        raise NotImplementedError()

    @abstractmethod
    def feature_importance(self) -> pd.DataFrame:
        """Feature importance of the model

        Raises:
            NotFittedError: If the model hasn't been fitted before

        Returns:
            pd.DataFrame: Feature importances
        """
        raise NotImplementedError()


class SKLearnTabularWrapper(SKLearnWrapper):
    def __init__(
        self,
        ag_predictor,
        x_w_columns: Optional[List] = None,
        hyperparameters: Optional[Union[List, Dict]] = None,
        presets: Optional[str] = "best_quality",
        ag_args_fit: Optional[Dict] = None,
        feature_generator: Optional[AbstractFeatureGenerator] = None,
        holdout_frac: Optional[float] = None,
    ):
        """Implementation of a SKLearnWrapper around a TabularPredictor

        Args:
            ag_predictor: TabularPredictor to be wrapped
            x_w_columns: Column names for the used DataFrame at training time
            hyperparameters: HyperParameters for the ag_predictor
            presets: Presets for the ag_predictor
            ag_args_fit: AutoGluon fit parameters for the ag_predictor
            feature_generator: AutoGluon feature generator for the ag_predictor
            holdout_frac: holdout_frac used in case of classification

        Raises:
            UnsupportedPredictorType: If the predictor is not a TabularPredictor
        """
        if not isinstance(ag_predictor, TabularPredictor):
            raise UnsupportedPredictorType()
        super().__init__(
            ag_predictor,
            x_w_columns,
            hyperparameters,
            presets,
            ag_args_fit,
            feature_generator,
            holdout_frac,
        )

    def set_sample_weight(self, sample_weight=None):
        self.ag_predictor.sample_weight = sample_weight

    def score(self, X, y, sample_weight=None) -> Dict[str, Any]:
        test_data = self._df_transformer.fit_transform(X)
        y_pred = self.ag_predictor.predict(test_data).values
        if self.ag_predictor.problem_type in ["binary", "multiclass"]:
            return {
                "accuracy": accuracy_score(y, y_pred, sample_weight=sample_weight),
                "y": y,
                "y_pred": y_pred,
                "sample_weight": sample_weight,
            }
        elif self.ag_predictor.problem_type == "regression":
            return {
                "r2": r2_score(y, y_pred, sample_weight=sample_weight),
                "y": y,
                "y_pred": y_pred,
                "sample_weight": sample_weight,
            }
        else:
            raise UnsupportedProblemType()

    def feature_importance(self) -> pd.DataFrame:
        if self.train_data is None:
            raise NotFittedError("The predictor needs to be fitted before")
        return self.ag_predictor.feature_importance(self.train_data)

    def add_output(self, train_data: pd.DataFrame, y) -> pd.DataFrame:
        train_data[self.ag_predictor.label] = y
        return train_data


class SKLearnMultilabelWrapper(SKLearnWrapper):
    def __init__(
        self,
        ag_predictor,
        x_w_columns: Optional[List] = None,
        hyperparameters: Optional[Union[List, Dict]] = None,
        presets: Optional[str] = "best_quality",
        ag_args_fit: Optional[Dict] = None,
        feature_generator: Optional[AbstractFeatureGenerator] = None,
        holdout_frac: Optional[float] = None,
    ):
        """Implementation of a SKLearnWrapper around a MultiLabelPredictor

        Args:
            ag_predictor: MultiLabelPredictor to be wrapped
            x_w_columns: Column names for the used DataFrame at training time
            hyperparameters: HyperParameters for each ag_predictors
            presets: Presets for each ag_predictors
            ag_args_fit: AutoGluon fit parameters for each ag_predictors
            feature_generator: AutoGluon feature generator for each ag_predictors
            holdout_frac: holdout_frac used in case of classification

        Raises:
            UnsupportedPredictorType: If the predictor is not a MultiLabelPredictor
        """
        if not isinstance(ag_predictor, MultilabelPredictor):
            raise UnsupportedPredictorType()
        super().__init__(
            ag_predictor,
            x_w_columns,
            hyperparameters,
            presets,
            ag_args_fit,
            feature_generator,
            holdout_frac,
        )

    def set_sample_weight(self, sample_weight=None):
        for label in self.ag_predictor.labels:
            self.ag_predictor.get_predictor(label=label).sample_weight = sample_weight

    def add_output(self, train_data, y):
        train_data[self.ag_predictor.labels] = pd.DataFrame(
            y, columns=self.ag_predictor.labels
        )
        return train_data

    def score(self, X, y, sample_weight=None) -> Dict[str, Any]:
        test_data = self._df_transformer.fit_transform(X)
        y_pred = self.ag_predictor.predict(test_data).values
        # FIXME: Problem type is uniformly the same
        # in our case. But to make wrapper reusable everywhere iterate over
        # y and have list of accuracies
        if self.ag_predictor.get_predictor(
            self.ag_predictor.labels[0]
        ).problem_type in ["binary", "multiclass"]:
            return {
                "accuracy": accuracy_score(y, y_pred, sample_weight=sample_weight),
                "y": y,
                "y_pred": y_pred,
                "sample_weight": sample_weight,
            }
        elif self.ag_predictor.get_predictor(
            self.ag_predictor.labels[0]
        ).problem_type in ["regression"]:
            return {
                "r2": r2_score(y, y_pred, sample_weight=sample_weight),
                "y": y,
                "y_pred": y_pred,
                "sample_weight": sample_weight,
            }
        else:
            raise UnsupportedProblemType()

    def feature_importance(self) -> pd.DataFrame:
        return super().feature_importance()
