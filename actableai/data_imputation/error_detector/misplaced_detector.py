from typing import List, Set

import pandas as pd

from actableai.data_imputation.config import logger
from actableai.data_imputation.error_detector.match_condition import ConditionOp
from actableai.data_imputation.error_detector.base_error_detector import (
    BaseErrorDetector,
)
from actableai.data_imputation.error_detector.cell_erros import ErrorType, CellError
from actableai.data_imputation.error_detector.column_format import (
    MatchRules,
    MatchStrRule,
    PresetRuleName,
)
from actableai.data_imputation.error_detector.error_detector import CellErrors
from actableai.data_imputation.meta import ColumnType
from actableai.data_imputation.meta.column import ColumnName
from actableai.data_imputation.type_recon.regex_consts import REGEX_CONSTS
from actableai.data_imputation.type_recon.type_detector import DfTypes


class MisplacedDetector(BaseErrorDetector):
    def __init__(
        self,
        *,
        preset_rules: List[PresetRuleName] = (),
        customize_rules: MatchRules = MatchRules([]),
    ):
        super(MisplacedDetector).__init__()
        self._rules = customize_rules
        self._preset_rules = preset_rules

    @property
    def mentioned_columns(self) -> Set[ColumnName]:
        return set([column_name for (column_name, _) in self._rules])

    def update_df(self, df: pd.DataFrame):
        self._df = df

    def setup(self, df: pd.DataFrame, dftypes: DfTypes):
        super(MisplacedDetector, self).setup(df, dftypes)
        for col in self._df.columns:
            col_type = self._dftypes[col]
            if (
                col_type == ColumnType.Percentage
                and PresetRuleName.SmartPercentage in self._preset_rules
            ):
                self._rules.append(
                    MatchStrRule(
                        column=col,
                        match_str=REGEX_CONSTS[ColumnType.Percentage],
                        is_regex=True,
                        op=ConditionOp.IQ,
                    )
                )
            elif (
                col_type == ColumnType.Temperature
                and PresetRuleName.SmartTemperature in self._preset_rules
            ):
                self._rules.append(
                    MatchStrRule(
                        column=col,
                        match_str=REGEX_CONSTS[ColumnType.Temperature],
                        is_regex=True,
                        op=ConditionOp.IQ,
                    )
                )

    def detect_cells(self) -> CellErrors:
        errors = CellErrors(self._dftypes)

        for col, match_rule in self._rules:
            logger.info(f"Detecting MISPLACED errors for {col}")

            series = self._df[col]
            misplaced_indexes = match_rule.find_misplaced(series)
            for index in misplaced_indexes:
                errors.append(CellError(col, index, ErrorType.MISPLACED))

        return errors
