from enum import Enum


class ParameterType(str, Enum):
    """Enum representing the different parameter types available."""

    VALUE = "value"
    LIST = "list"
    OPTIONS = "options"
    PARAMETERS = "parameters"


class ValueType(str, Enum):
    """
    TODO write documentation
    """

    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
