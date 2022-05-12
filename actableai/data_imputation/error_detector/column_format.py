import pandas as pd
import re
from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Enum
from pandas import Series
from typing import Text, List, Tuple, Iterator, Union

from actableai.data_imputation.config import logger
from actableai.data_imputation.error_detector.match_condition import ConditionOp
from actableai.data_imputation.meta.column import ColumnName, NumWithTagColumnMeta
from actableai.data_imputation.meta.types import ColumnType
from actableai.data_imputation.type_recon.type_detector import DfTypes


class PresetRuleName(Enum):
    SmartPercentage = "SmartPercentage"
    SmartTemperature = "SmartTemperature"


@dataclass(frozen=True)
class _MatchRule(ABC):
    column: ColumnName
    op: ConditionOp

    @abstractmethod
    def find_misplaced(self, series: Series) -> Iterator[int]:
        raise NotImplementedError


@dataclass(frozen=True)
class MatchStrRule(_MatchRule):
    match_str: Text
    is_regex: bool

    def find_misplaced(self, series: Series) -> Iterator[int]:
        if self.op not in [ConditionOp.EQ, ConditionOp.IQ]:
            logger.warning(
                f"op {self.op} does not support in MatchStrRule, ignore the rule"
            )
            return []

        if str(series.dtype).startswith("float"):
            series_str = series.apply(lambda x: str(int(x)) if x // 1 == x else str(x))
        else:
            series_str = series.apply(lambda x: str(x))

        if self.is_regex:
            match_result = series_str.str.match(f"^{self.match_str}$")
        else:
            match_result = series_str.apply(lambda x: self.match_str == x)

        for index in match_result.index:
            match = match_result[index] and not pd.isna(series[index])
            if self.op is ConditionOp.EQ:
                if match:
                    yield index
            elif self.op is ConditionOp.IQ:
                if not match:
                    yield index
            else:
                raise NotImplementedError


@dataclass(frozen=True)
class MatchNumRule(_MatchRule):
    match_val: Union[int, float]

    def find_misplaced(self, series: Series) -> Iterator[int]:
        dtype = str(series.dtype)
        if not dtype.startswith("float") and not dtype.startswith("int"):
            logger.debug(
                f"series dtype {dtype} does not support in MatchNumRule, ignore the rule"
            )
            return []

        if self.op == ConditionOp.EQ:
            match_result = series.where(lambda x: x == self.match_val).dropna()
        elif self.op == ConditionOp.IQ:
            match_result = series.where(lambda x: x != self.match_val).dropna()
        elif self.op == ConditionOp.LTE:
            match_result = series.where(lambda x: x <= self.match_val).dropna()
        elif self.op == ConditionOp.GTE:
            match_result = series.where(lambda x: x >= self.match_val).dropna()
        elif self.op == ConditionOp.LT:
            match_result = series.where(lambda x: x < self.match_val).dropna()
        elif self.op == ConditionOp.GT:
            match_result = series.where(lambda x: x > self.match_val).dropna()
        else:
            raise NotImplementedError

        for index in match_result.index:
            yield index


@dataclass(repr=False, frozen=True)
class MatchRule:
    column: Text
    value: Text
    op: ConditionOp

    def __str__(self):
        return f"{self.column}{self.op.value}{self.value}"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def parse(
        cls, dftypes: DfTypes, condition_string: Text
    ) -> Union[MatchStrRule, MatchNumRule, type(NotImplemented)]:
        delimiter_regex = "|".join([rf"\s*{v.value}" for v in ConditionOp])
        delimiter = re.findall(delimiter_regex, condition_string)
        if len(delimiter) != 1:
            return NotImplemented

        condition_split = re.split(delimiter_regex, condition_string)
        if len(condition_split) != 2:
            return NotImplemented

        column = condition_split[0].strip()
        value = condition_split[1].strip()
        delimiter = ConditionOp(delimiter[0].strip())
        if dftypes[column] in [
            ColumnType.Integer,
            ColumnType.Float,
        ]:
            try:
                return MatchNumRule(
                    column=column,
                    match_val=float(value),
                    op=delimiter,
                )
            except ValueError:
                return NotImplemented
        elif dftypes[column] in [
            ColumnType.NumWithTag,
            ColumnType.Percentage,
            ColumnType.Temperature,
        ]:
            meta = dftypes.get_meta(column)
            if isinstance(meta, NumWithTagColumnMeta):
                try:
                    return MatchNumRule(
                        column=meta.get_num_column_name(),
                        match_val=float(value),
                        op=delimiter,
                    )
                except ValueError:
                    return NotImplemented
            else:
                raise NotImplementedError("Should not reach here")
        else:
            is_regex = value.startswith("s/") and value.endswith("/g")
            if is_regex:
                match_str = value[2:-2]
            else:
                match_str = value
            if delimiter not in [ConditionOp.EQ, ConditionOp.IQ]:
                return NotImplemented
            return MatchStrRule(
                column=column,
                match_str=match_str,
                is_regex=is_regex,
                op=delimiter,
            )


@dataclass(repr=False, frozen=True)
class MatchRuleGroup:
    conditions: List[MatchRule]

    def __str__(self):
        return "&".join(map(str, self.conditions))

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        for c in self.conditions:
            yield c


class MatchRules:
    def __init__(self, expect_formats: List[_MatchRule]):
        self._expect_formats = {
            expect_format.column: expect_format for expect_format in expect_formats
        }

    def __iter__(self) -> Iterator[Tuple[ColumnName, _MatchRule]]:
        for col in self._expect_formats:
            yield col, self._expect_formats[col]

    def append(self, rule: _MatchRule):
        self._expect_formats[rule.column] = rule

    @classmethod
    def parse(cls, dtypes, condition_string: Text) -> "MatchRules":
        return cls(
            list(
                filter(
                    lambda x: x is not NotImplemented,
                    [
                        MatchRule.parse(dtypes, condition.strip())
                        for condition in condition_string.split("OR")
                    ],
                )
            )
        )
