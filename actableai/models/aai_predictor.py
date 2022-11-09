class AAIModel:
    model_version = 3

    def __init__(self) -> None:
        self.version = self.model_version


class AAITabularModel(AAIModel):
    def __init__(self, predictor) -> None:
        super().__init__()
        self.predictor = predictor


class AAIInterventionalModel(AAIModel):
    def __init__(self, intervention_predictor) -> None:
        super().__init__()
        self.intervention_predictor = intervention_predictor


class AAITabularModelInterventional(AAITabularModel):
    def __init__(self, predictor, intervention_model) -> None:
        super().__init__(predictor)
        self.intervention_model = intervention_model
