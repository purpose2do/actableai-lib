import shap
import numpy as np


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
        model = self.autogluon_predictor._trainer.load_model(model_name).model

        problem_type = self.autogluon_predictor.problem_type
        self.is_classification = (
            problem_type == "multiclass" or problem_type == "binary"
        )

        self.explainer = shap.TreeExplainer(model)

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
        transformed_data = self.autogluon_predictor.transform_features(data)
        shap_values = self.explainer.shap_values(transformed_data, *args, **kwargs)

        if not self.is_classification:
            return shap_values

        shap_values = np.array(shap_values)

        pred = self.autogluon_predictor.predict_proba(data, as_pandas=False)

        row_index, column_index = np.meshgrid(
            np.arange(shap_values.shape[1]), np.arange(shap_values.shape[2])
        )
        return shap_values[pred.argmax(axis=1), row_index, column_index].transpose(1, 0)
