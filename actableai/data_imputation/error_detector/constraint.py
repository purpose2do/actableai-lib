import re
from dataclasses import dataclass
from typing import Text, List, Set

from actableai.data_imputation.error_detector.match_condition import ConditionOp


@dataclass(repr=False, frozen=True)
class Condition:
    col1: Text
    col2: Text
    condition: ConditionOp

    def __str__(self):
        return f"{self.col1}{self.condition.value}{self.col2}"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def parse(cls, condition_string: Text) -> "Condition":
        delimiter_regex = "|".join([rf"\s*{v.value}" for v in ConditionOp])
        delimiter = re.findall(delimiter_regex, condition_string)
        if len(delimiter) != 1:
            return NotImplemented

        condition_split = re.split(delimiter_regex, condition_string)
        if len(condition_split) != 2:
            return NotImplemented

        condition_a = condition_split[0].strip()
        condition_b = condition_split[1].strip()

        return Condition(
            condition_a,
            condition_b,
            ConditionOp(delimiter[0].strip()),
        )

    @property
    def mentioned_columns(self) -> Set[Text]:
        return {self.col1, self.col2}


@dataclass(repr=False, frozen=True)
class ConditionGroup:
    conditions: List[Condition]

    def __str__(self):
        return "&".join(map(str, self.conditions))

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        for c in self.conditions:
            yield c

    @classmethod
    def parse(cls, condition_string: Text) -> "ConditionGroup":
        return cls(
            list(
                filter(
                    lambda x: x is not NotImplemented,
                    [
                        Condition.parse(condition)
                        for condition in condition_string.split("&")
                    ],
                )
            )
        )

    @property
    def mentioned_columns(self) -> Set[Text]:
        columns = set()
        for condition in self.conditions:
            columns = columns.union(condition.mentioned_columns)
        return columns


@dataclass(repr=False, frozen=True)
class Constraint:
    """
    <when> condition match, then the data match <then> condition as invalid
    """

    when: ConditionGroup
    then: ConditionGroup

    def __str__(self):
        return f"{str(self.when)} -> {str(self.then)}"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def parse(cls, constrain_string: Text) -> "Constraint":
        constrain_split = re.split(r"\s*->\s*", constrain_string)
        if len(constrain_split) != 2:
            return NotImplemented

        when = ConditionGroup.parse(constrain_split[0])
        if when is NotImplemented:
            return NotImplemented
        then = ConditionGroup.parse(constrain_split[1])
        if then is NotImplemented:
            return NotImplemented

        return cls(when, then)

    @property
    def mentioned_columns(self) -> Set[Text]:
        return self.when.mentioned_columns.union(self.then.mentioned_columns)


@dataclass(repr=False, frozen=True)
class Constraints:
    constraints: List[Constraint]

    def __str__(self):
        return " OR ".join(map(str, self.constraints))

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        for c in self.constraints:
            yield c

    @classmethod
    def parse(cls, constraints_string: Text) -> "Constraints":
        constraints = re.split(r"\s*OR\s*|\s*\n\s*", constraints_string)
        return cls(
            list(
                filter(
                    lambda x: x is not NotImplemented,
                    [Constraint.parse(constrain) for constrain in constraints],
                )
            )
        )

    @property
    def mentioned_columns(self) -> Set[Text]:
        columns = set()
        for constraint in self.constraints:
            columns = columns.union(constraint.mentioned_columns)
        return columns

    def extends(self, constraints: "Constraints"):
        self.constraints.extend(constraints.constraints)
