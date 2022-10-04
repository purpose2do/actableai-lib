from actableai.models.config import MODEL_DEPLOYMENT_VERSION


class AAIModel:
    model_version = MODEL_DEPLOYMENT_VERSION

    def __init__(self, version: int = model_version) -> None:
        self.version = version


class AAITabularModel(AAIModel):
    def __init__(
        self,
        version: int,
        predictor,
    ) -> None:
        super().__init__(version)
        self.predictor = predictor


class AAITabularModelInterventional(AAITabularModel):
    def __init__(
        self,
        version: int,
        predictor,
        intervention_model,
    ) -> None:
        super().__init__(
            version,
            predictor,
        )
        self.intervention_model = intervention_model


class AAIInterventionalModel(AAIModel):
    def __init__(self, version: int, intervention_predictor) -> None:
        super().__init__(version)
        self.intervention_predictor = intervention_predictor
