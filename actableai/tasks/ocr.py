from typing import Any, Dict, Optional, Iterable

import pandas as pd
from PIL.Image import Image as ImageType

from actableai.parameters.options import OptionsParameter
from actableai.parameters.parameters import Parameters
from actableai.tasks.base import AAITask
from actableai.tasks import TaskType


class AAIOCRTask(AAITask):
    @staticmethod
    def _get_ocr_model_parameters() -> OptionsParameter[Parameters]:
        from actableai.ocr.models.base import Model
        from actableai.ocr.models import model_dict

        available_models = [
            Model.tesseract,
        ]
        default_model = Model.tesseract

        options = {}
        for model in available_models:
            model_parameters = model_dict[model].get_parameters()
            options[model] = {
                "display_name": model_parameters.display_name,
                "value": model_parameters,
            }

        return OptionsParameter[Parameters](
            name="ocr_model",
            display_name="OCR Model",
            description="Model used for character recognition",
            is_multi=False,
            default=default_model,
            options=options,
        )

    @classmethod
    def get_parameters(cls) -> Parameters:
        parameters = [
            cls._get_ocr_model_parameters(),
        ]

        return Parameters(
            name="ocr_parameters",
            display_name="OCR Parameters",
            parameters=parameters,
        )

    @AAITask.run_with_ray_remote(TaskType.OCR)
    def run(
        self,
        images: Iterable[ImageType],
        parameters: Optional[Dict[str, Any]] = None,
    ):
        import time
        from actableai.data_validation.base import CheckLevels
        from actableai.ocr.models import model_dict

        start = time.time()

        parameters_validation = None
        parameters_definition = self.get_parameters()
        if parameters is None or len(parameters) <= 0:
            parameters = parameters_definition.get_default()
        else:
            (
                parameters_validation,
                parameters,
            ) = parameters_definition.validate_process_parameter(parameters)

        data_validation_results = []

        if parameters_validation is not None:
            data_validation_results += parameters_validation.to_check_results(
                name="Parameters"
            )

        failed_checks = [x for x in data_validation_results if x is not None]
        if CheckLevels.CRITICAL in [x.level for x in failed_checks]:
            return {
                "status": "FAILURE",
                "validations": [
                    {"name": x.name, "level": x.level, "message": x.message}
                    for x in failed_checks
                ],
                "runtime": time.time() - start,
                "data": {},
            }

        ocr_model_name, ocr_model_parameters = next(
            iter(parameters["ocr_model"].items())
        )
        ocr_model_class = model_dict[ocr_model_name]
        ocr_model = ocr_model_class(
            parameters=ocr_model_parameters,
            process_parameters=False,
        )

        parsed_text = ocr_model.transform(data=images)
        df_text = pd.DataFrame(
            {
                "parsed_text": parsed_text,
            }
        )

        return {
            "data": {
                "parsed_text": df_text,
            },
            "status": "SUCCESS",
            "messenger": "",
            "runtime": time.time() - start,
            "validations": [
                {"name": x.name, "level": x.level, "message": x.message}
                for x in failed_checks
            ],
        }
