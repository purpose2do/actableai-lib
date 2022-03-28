import pytest
import ray

from actableai.utils.testing import init_ray as _init_ray


@pytest.fixture(scope="function")
def init_ray():
    if not ray.is_initialized():
        _init_ray()

        yield None

        ray.shutdown()
    else:
        yield None
