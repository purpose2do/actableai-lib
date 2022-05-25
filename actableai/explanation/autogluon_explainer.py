import pandas as pd
import shap

from autogluon.features.generators import DatetimeFeatureGenerator


class AutoGluonShapTreeExplainer:
    """
    TODO write documentation
    """

    def __init__(self, autogluon_predictor):
        """
        TODO write documentation
        """
        self.autogluon_predictor = autogluon_predictor
        model_name = self.autogluon_predictor.get_model_best()

        self.explainer = shap.TreeExplainer(
            self.autogluon_predictor._trainer.load_model(model_name).model
        )

    @staticmethod
    def is_predictor_compatible(autogluon_predictor):
        """
        TODO write documentation
        """
        model_name = autogluon_predictor.get_model_best()
        try:
            shap.TreeExplainer(
                autogluon_predictor._trainer.load_model(model_name).model
            )
        except:
            return False
        return True

    def shap_values(self, data, *args, **kwargs):
        """
        TODO write documentation
        """
        data = self.autogluon_predictor.transform_features(data)
        return self.explainer.shap_values(data, *args, **kwargs)
