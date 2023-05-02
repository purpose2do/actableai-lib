from typing import Union, Dict, Any, Optional, List, Set

from enum import Enum

from ray.serve.config import AutoscalingConfig
from PIL.Image import Image as ImageType

from actableai.ocr.serve.base import BaseOCRServe


class TesseractModelType(str, Enum):
    fast = "fast"
    normal = "normal"
    best = "best"


class TesseractServe(BaseOCRServe):
    @classmethod
    def deploy(
        cls,
        *,
        ray_autoscaling_configs: Union[Dict, AutoscalingConfig],
        ray_options: Dict[str, Any],
        tessdata_path: Optional[str] = None,
        tessdata_fast_path: Optional[str] = None,
        tessdata_best_path: Optional[str] = None,
    ):
        return cls._deploy(
            ray_autoscaling_configs=ray_autoscaling_configs,
            ray_options=ray_options,
            init_kwargs={
                "tessdata_path": tessdata_path,
                "tessdata_fast_path": tessdata_fast_path,
                "tessdata_best_path": tessdata_best_path,
            },
        )

    def __init__(
        self,
        tessdata_path: Optional[str] = None,
        tessdata_fast_path: Optional[str] = None,
        tessdata_best_path: Optional[str] = None,
    ):
        # TODO maybe check validity of the paths
        self.model_type_paths = {}
        if tessdata_path is not None:
            self.model_type_paths[TesseractModelType.normal] = tessdata_path
        if tessdata_fast_path is not None:
            self.model_type_paths[TesseractModelType.fast] = tessdata_fast_path
        if tessdata_best_path is not None:
            self.model_type_paths[TesseractModelType.best] = tessdata_best_path

    def _get_config(
        self,
        model_type: Optional[TesseractModelType] = None,
        page_segmentation_mode: int = 3,
    ) -> str:
        if model_type is None:
            return ""
        if model_type not in self.model_type_paths:
            raise ValueError(f"Model type not supported: {model_type}")

        return f"--tessdata-dir {self.model_type_paths[model_type]} --psm {page_segmentation_mode}"

    def get_available_languages(
        self,
        model_type: TesseractModelType = TesseractModelType.normal,
        page_segmentation_mode: int = 3,
    ) -> Set[str]:
        import pytesseract

        config = self._get_config(
            model_type,
            page_segmentation_mode=page_segmentation_mode,
        )
        return set(pytesseract.get_languages(config=config))

    def transform(
        self,
        image: ImageType,
        model_type: Optional[TesseractModelType] = None,
        language_list: Union[List[str], str] = "eng",
        page_segmentation_mode: int = 3,
    ) -> str:
        import pytesseract

        if not isinstance(language_list, list):
            language_list = [language_list]

        config = self._get_config(
            model_type=model_type,
            page_segmentation_mode=page_segmentation_mode,
        )
        available_languages = self.get_available_languages(
            model_type=model_type,
            page_segmentation_mode=page_segmentation_mode,
        )

        for language in language_list:
            if language not in available_languages:
                raise ValueError(f"Invalid language: {language}")

        lang = "+".join(language_list)

        return pytesseract.image_to_string(image=image, lang=lang, config=config)
