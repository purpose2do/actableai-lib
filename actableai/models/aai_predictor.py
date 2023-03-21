from actableai.parameters.numeric import IntegerParameter, FloatParameter
from actableai.parameters.options import OptionsParameter
from actableai.parameters.parameters import Parameters
from actableai.parameters.string import StringParameter


class AAIModel:
    model_version = 4

    def __init__(self) -> None:
        self.version = self.model_version


class AAITabularModel(AAIModel):
    @staticmethod
    def _autogluon_feature_to_parameter(feature_name, feature_type, df_training):
        """
        TODO write documentation
        """

        if feature_type == "int":
            return IntegerParameter(
                name=feature_name,
                display_name=feature_name,
            )
        if feature_type == "float":
            return FloatParameter(
                name=feature_name,
                display_name=feature_name,
            )
        if feature_type == "category":
            available_categories = (
                df_training[feature_name].dropna().astype(str).unique()
            )

            return OptionsParameter[str](
                name=feature_name,
                display_name=feature_name,
                is_multi=False,
                options={
                    category: {"display_name": category, "value": category}
                    for category in available_categories
                },
            )
        if feature_type == "datetime":
            # TODO datetime parameter
            pass

        return StringParameter(
            name=feature_name,
            display_name=feature_name,
        )

    def __init__(self, predictor, df_training, explainer=None):
        super().__init__()
        self.predictor = predictor

        features_map = predictor.feature_metadata.type_map_raw
        self.feature_parameters = Parameters(
            name="feature_parameters",
            display_name="feature_parameters",
            parameters=[
                self._autogluon_feature_to_parameter(
                    feature_name=feature_name,
                    feature_type=feature_type,
                    df_training=df_training,
                )
                for feature_name, feature_type in features_map.items()
            ],
        ).dict()

        self.explainer = explainer


class AAIInterventionalModel(AAIModel):
    def __init__(self, intervention_predictor) -> None:
        super().__init__()
        self.intervention_predictor = intervention_predictor


class AAITabularModelInterventional(AAITabularModel):
    def __init__(self, predictor, intervention_model, df_training) -> None:
        super().__init__(predictor, df_training)
        self.intervention_model = intervention_model
