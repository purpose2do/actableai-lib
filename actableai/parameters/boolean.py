from typing import Dict

from actableai.parameters.base import BaseParameter
from actableai.parameters.options import OptionsSpace, Option
from actableai.parameters.type import ParameterType


class BooleanParameter(BaseParameter):
    """Simple Boolean parameter."""

    parameter_type: ParameterType = ParameterType.BOOL
    default: bool


class BooleanSpace(OptionsSpace[bool]):
    """Boolean space parameter."""

    options: Dict[bool, Option[bool]] = {
        "true": Option(display_name="True", value=True),
        "false": Option(display_name="False", value=False),
    }
