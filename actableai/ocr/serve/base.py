from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional, Tuple

from ray.serve.config import AutoscalingConfig


class BaseOCRServe(ABC):
    @classmethod
    def _deploy(
        cls,
        *,
        ray_autoscaling_configs: Union[Dict, AutoscalingConfig],
        ray_options: Dict[str, Any],
        init_args: Optional[Tuple[Any]] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
    ):
        from ray import serve

        return serve.deployment(
            cls,
            name=cls.__name__,
            autoscaling_config=ray_autoscaling_configs,
            ray_actor_options=ray_options,
            init_args=init_args,
            init_kwargs=init_kwargs,
        ).deploy()

    @classmethod
    def get_deployment(cls):
        """
        TODO write documentation
        """
        from ray import serve

        return serve.get_deployment(cls.__name__)

    @classmethod
    def get_handle(cls):
        """
        TODO write documentation
        """
        return cls.get_deployment().get_handle()

    @abstractmethod
    def transform(self, *args, **kwargs):
        pass
