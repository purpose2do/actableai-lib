import pytest

from autogluon.tabular.predictor import TabularPredictor
from autogluon.tabular.configs.hyperparameter_configs import (
    hyperparameter_config_dict,
)

from actableai.explanation.autogluon_explainer import AutoGluonShapTreeExplainer
from actableai.utils import explanation_hyperparameters
from actableai.utils.dataset_generator import DatasetGenerator


class TestAutoGluonTreeExplainer:
    @pytest.mark.parametrize("problem_type", ["regression", "multiclass", "binary"])
    @pytest.mark.parametrize("model", explanation_hyperparameters().keys())
    def test_shap_values_regression(self, problem_type, model, tmp_path):
        explain_hyperparameters = explanation_hyperparameters()
        hyperparameters = {model: explain_hyperparameters[model]}

        df = DatasetGenerator.generate(
            columns_parameters=[
                {"type": "number", "float": False},
                {"type": "number", "float": True},
                {"type": "date"},
                {"type": "text", "n_categories": 3},
                {"type": "text", "word_range": (1, 2)},
                {"type": "text", "word_range": (5, 10)},
                {"name": "target_regression", "type": "number", "float": True},
                {"name": "target_multiclass", "type": "text", "n_categories": 3},
                {"name": "target_binary", "type": "text", "n_categories": 2},
            ],
            rows=20,
            random_state=0,
        )

        predictor = TabularPredictor(
            label=f"target_{problem_type}",
            path=tmp_path,
            problem_type=problem_type
        )

        try:
            predictor = predictor.fit(
                train_data=df,
                presets="medium_quality_faster_train",
                hyperparameters=hyperparameters
            )
        except ValueError:
            return

        while not AutoGluonShapTreeExplainer.is_predictor_compatible(predictor):
            predictor.delete_models(
                models_to_delete=predictor.get_model_best(),
                dry_run=False,
                allow_delete_cascade=True,
            )

        explainer = AutoGluonShapTreeExplainer(predictor)
        shap_values = explainer.shap_values(df)

        assert shap_values is not None
        assert len(shap_values.shape) == 2
        assert shap_values.shape[0] == df.shape[0]
        assert shap_values.shape[1] == df.shape[1]
