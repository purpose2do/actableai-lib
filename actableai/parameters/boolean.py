from typing import Dict

from actableai.parameters.list import ListParameter
from actableai.parameters.options import OptionsSpace, Option
from actableai.parameters.value import ValueParameter

BooleanParameter = ValueParameter[bool]
BooleanListParameter = ListParameter[bool]


class BooleanSpace(OptionsSpace[bool]):
    """Boolean space parameter."""

    options: Dict[bool, Option[bool]] = {
        "true": Option(display_name="True", value=True),
        "false": Option(display_name="False", value=False),
    }
