from dataclasses import dataclass

from actableai.data_imputation.error_detector.column_format import (
    MatchRules,
)
from actableai.data_imputation.error_detector.constraint import (
    Constraints,
)
from actableai.data_imputation.type_recon.type_detector import DfTypes


@dataclass
class RulesRaw:
    validations: str
    misplaced: str


class RulesBuilder:
    def __init__(self, constraints: Constraints, match_rules: MatchRules):
        self._constraints = constraints
        self._match_rules = match_rules

    @property
    def constraints(self):
        return self._constraints

    @property
    def match_rules(self):
        return self._match_rules

    @classmethod
    def parse(cls, dftypes: DfTypes, rules: RulesRaw) -> "RulesBuilder":
        return cls(
            constraints=Constraints.parse(rules.validations),
            match_rules=MatchRules.parse(dftypes, rules.misplaced),
        )
