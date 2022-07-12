from typing import Optional, List, Dict, Any

from pydantic import BaseModel

from actableai.parameters.parameters import BaseParameter
from actableai.parameters.spaces import OptionsSpace


class ModelParameters(BaseModel):
    """
    TODO write documentation
    """

    name: str
    display_name: str
    description: Optional[str]
    parameters: List[BaseParameter]


ModelSpace = OptionsSpace[ModelParameters]
