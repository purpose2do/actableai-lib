import logging
from collections import defaultdict
from string import Template
from typing import List, Iterable, Set

import numpy as np
import pandas as pd
import pandasql as ps

from actableai.data_imputation.config import (
    logger,
    TYPO_APPEARANCE_FREQUENCY_THRESHOLD,
)
from actableai.data_imputation.error_detector.cell_erros import (
    ErrorCandidate,
    ErrorColumns,
    CellError,
    ErrorType,
)
from actableai.data_imputation.error_detector.constraint import (
    Constraints,
    Constraint,
)
from actableai.data_imputation.meta.column import ColumnName


class ValidationConstrainsChecker:
    def __init__(self, constraints: Constraints):
        self._constraints = constraints

    @property
    def mentioned_columns(self) -> Set[ColumnName]:
        return self._constraints.mentioned_columns

    @staticmethod
    def _find_ids_invalid(constrain: Constraint, df: pd.DataFrame) -> Iterable[int]:
        conditions = [
            f"t1.`{condition.col1}`{condition.condition.value}t2.`{condition.col2}`"
            for condition in constrain.when
        ] + [
            f"t1.`{condition.col1}`{condition.condition.value}t2.`{condition.col2}`"
            for condition in constrain.then
        ]

        template = Template(
            "SELECT DISTINCT t1.id FROM df as t1, df as t2 "
            "WHERE t1.id<>t2.id AND $cond"
        )

        sql = template.substitute(cond=" AND ".join(conditions))
        logger.debug(f"Running SQL for ValidationDetector: {sql}")
        results = ps.sqldf(
            sql,
            {"df": df},
        )
        for row in results.itertuples(index=False):
            yield row.id

    @staticmethod
    def _pick_wrong_index(
        df: pd.DataFrame, all_candidates: List[ErrorCandidate]
    ) -> Set[int]:
        # count the time where values for multiple columns come together
        statistics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for candidate in all_candidates:
            index = candidate.index
            from_cols = candidate.columns.columns_from_when
            for to_col in candidate.columns.columns_from_then:
                concat_str = ",".join(
                    map(
                        str,
                        [
                            df.at[index, col]
                            for col in from_cols
                            if not pd.isna(df.at[index, col])
                        ],
                    )
                )

                statistics[to_col][concat_str][
                    str(df.at[candidate.index, to_col])
                ].append(index)

        wrong_index = set()
        # find the index of the dataframe with: cell value count in minority or only appear once
        for statistics_in_group in statistics.values():
            for count_group in statistics_in_group.values():
                sum_count = float(sum(map(len, count_group.values())))
                for index_list in count_group.values():
                    if (
                        len(index_list) == 1
                        or len(index_list) / sum_count
                        < TYPO_APPEARANCE_FREQUENCY_THRESHOLD
                    ):
                        wrong_index = wrong_index.union(set(index_list))

        if len(wrong_index) == 0:
            logging.warning(
                "Unfortunately, we are not able to find the wrong index, the rule might not make sense to the dataset"
            )
        return wrong_index

    def find_all_unmatched(self, df: pd.DataFrame) -> List[ErrorCandidate]:
        df["id"] = range(len(df))

        columns_from_when_group = defaultdict(
            list
        )  # row id to list of nest list of column names
        columns_from_then_group = defaultdict(
            list
        )  # row id to list of nest list of column names
        for constrain in self._constraints:
            for row_id in self._find_ids_invalid(constrain, df):
                columns_from_when_group[row_id].extend(
                    [[cond.col1, cond.col2] for cond in constrain.when]
                )
                columns_from_then_group[row_id].extend(
                    [[cond.col1, cond.col2] for cond in constrain.then],
                )

        df.drop("id", inplace=True, axis=1)
        all_candidates = []
        for index in columns_from_when_group:
            all_candidates.append(
                ErrorCandidate(
                    index=index,
                    columns=ErrorColumns(
                        set(
                            sum(
                                columns_from_when_group[index],
                                [],
                            )
                        ),
                        set(
                            sum(
                                columns_from_then_group[index],
                                [],
                            )
                        ),
                    ),
                )
            )
        return all_candidates

    def detect_most_possible_candidates(self, df: pd.DataFrame) -> List[ErrorCandidate]:
        all_candidates = self.find_all_unmatched(df)
        wrong_index = self._pick_wrong_index(df, all_candidates)
        picked_candidates = []
        for candidate in all_candidates:
            if candidate.index in wrong_index:
                picked_candidates.append(candidate)
        return picked_candidates

    def detect_typo_cells(self, df: pd.DataFrame) -> List[CellError]:
        all_candidates = self.find_all_unmatched(df)
        logger.info(
            f"Found {len(all_candidates)} candidates, trim them using statistic methods"
        )

        wrong_index = self._pick_wrong_index(df, all_candidates)

        errors = []
        # select columns out of the possible columns
        for candidate in all_candidates:
            index = candidate.index
            if index not in wrong_index:
                continue

            potential_columns = list(candidate.potential_columns)
            counts = []
            for potential_column in potential_columns:
                other_columns = list(
                    filter(
                        lambda x: x != potential_column,
                        self._constraints.mentioned_columns,
                    )
                )
                combined_value = [(col, df.at[index, col]) for col in other_columns]
                select = [True] * len(df)
                for col, value in combined_value:
                    select &= df[col] == value
                df_select = df[select]
                counts.append(len(df_select))

            counts = np.asarray(counts)

            # because we know this is the wrong index,
            # so when apart from current column,
            # if the combined value for other columns (mentioned in constraints)
            # occur a lot compare use current column to combine with others
            # this column might be the wrong value
            max_count_index_list = np.flatnonzero(counts == np.max(counts))

            for max_count_index in max_count_index_list:
                errors.append(
                    CellError(
                        potential_columns[max_count_index],
                        index,
                        ErrorType.TYPO,
                    )
                )

        return errors
