from enum import Enum


class ConditionOp(Enum):
    IQ = "<>"
    LTE = "<="
    GTE = ">="
    EQ = "="
    LT = "<"
    GT = ">"
