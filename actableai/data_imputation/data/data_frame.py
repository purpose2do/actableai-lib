from typing import Optional, Dict, Set

import pandas as pd

from actableai.data_imputation.auto_fixer import EmptyTrainDataException
from actableai.data_imputation.auto_fixer.fix_info import FixInfoList
from actableai.data_imputation.auto_fixer.helper import (
    finalize_columns,
    fulfil_fix_back,
    merge_num_with_tag_columns,
)
from actableai.data_imputation.auto_fixer.misplaced_refiner import MisplacedRefiner
from actableai.data_imputation.auto_fixer.validation_refiner import ValidationRefiner
from actableai.data_imputation.auto_fixer.strategy import (
    get_fixer,
    get_quick_fixer_for_debug,
    FixStrategy,
    determine_fix_strategy,
)
from actableai.data_imputation.config import logger
from actableai.data_imputation.data.loader import LoaderType, load_data
from actableai.data_imputation.error_detector import ErrorDetector, CellErrors
from actableai.data_imputation.error_detector.base_error_detector import (
    BaseErrorDetector,
)
from actableai.data_imputation.meta.column import ColumnName
from actableai.data_imputation.meta.types import ColumnType
from actableai.data_imputation.processor import ProcessOps, Processor
from actableai.data_imputation.type_recon import TypeDetector
from actableai.data_imputation.type_recon.type_detector import DfTypes
from actableai.data_imputation.utils.memory import get_memory_usage


def _construct_new_df(
    df: pd.DataFrame,
    processor: Processor,
    error_detector: ErrorDetector,
    fix_strategy: Dict[ColumnName, FixStrategy],
) -> "DataFrame":
    new_df = DataFrame(df)
    new_df._cached_column_types = processor.get_column_types()
    new_df._error_detector = error_detector
    new_df._processor = processor
    new_df._fix_strategy = fix_strategy

    return new_df


class DataFrame(pd.DataFrame):
    def __init__(self, d: LoaderType):
        df = load_data(d)
        super().__init__(df)
        self._cached_column_types: Optional[DfTypes] = None
        self._error_detector: Optional[ErrorDetector] = None
        self._processor = Processor(self, self.column_types)
        self._fix_strategy = {col: FixStrategy.UNDECIDED for col in df.columns}
        self._all_fix_info: Optional[FixInfoList] = None
        self.__debug_mode = False

    @classmethod
    def from_dict(cls, data, orient="columns", dtype=None, columns=None) -> "DataFrame":
        return cls(pd.DataFrame.from_dict(data, orient, dtype, columns))

    @property
    def column_types(self) -> DfTypes:
        self.__detect_column_types_if_not_present()
        return self._cached_column_types

    @property
    def possible_column_types(self) -> Dict[ColumnName, Set[ColumnType]]:
        td = TypeDetector()
        return td.detect_possible_types(self)

    @property
    def fix_strategy(self):
        return {col: self._fix_strategy[col] for col in self._fix_strategy}

    @property
    def fix_info(self):
        return FixInfoList(list(self._all_fix_info))

    def enable_debug(self, enable: bool = True):
        self.__debug_mode = enable

    def __detect_column_types_if_not_present(self):
        if self._cached_column_types is None:
            td = TypeDetector()
            self._cached_column_types = td.detect(self)

    def override_column_type(self, column: ColumnName, column_type: ColumnType):
        self.__detect_column_types_if_not_present()
        self._cached_column_types.override(column, column_type)

    def _preprocess(self, errors: CellErrors) -> pd.DataFrame:
        self._processor.chain(
            [
                ProcessOps.EXCLUDE_UNSUPPORTED_COLUMNS,
                ProcessOps.EXPEND_NUM_WITH_TAG,
                ProcessOps.COLUMN_AS_DETECTED_TYPE_TO_TRAIN,
                ProcessOps.CATEGORY_TO_LABEL_NUMBER,
                ProcessOps.REPLACE_ALL_ERROR_TO_NA,
            ],
            errors,
        )
        preprocess_df = self._processor.get_processed_df()
        df = _construct_new_df(
            preprocess_df,
            self._processor,
            self._error_detector,
            self._fix_strategy,
        )
        return df

    def detect_error(self, *detectors: BaseErrorDetector) -> CellErrors:
        # expand columns to support numeric compare for NumWithTag types
        processor = Processor(self, self.column_types)
        expand_df = processor.expand_num_with_tag()

        self._error_detector = ErrorDetector()
        if len(detectors):
            self._error_detector.set_detectors(*detectors)
        return self._error_detector.detect_error(
            self, self.column_types, expand_df, processor.get_column_types()
        )

    def _gather_fix_info(
        self, errors: CellErrors, preprocessed_df: pd.DataFrame
    ) -> FixInfoList:
        all_fix_info = FixInfoList()
        for col in errors.columns:
            get_memory_usage("before column training")
            logger.info(f"Start fixing column: {col}, deciding fix strategy")
            fix_strategy = determine_fix_strategy(
                preprocessed_df[col], self.column_types[col], errors[col]
            )
            logger.info(f"Fixing column: {col} use strategy {fix_strategy}....")
            self._fix_strategy[col] = fix_strategy
            if fix_strategy == FixStrategy.UNABLE_TO_FIX:
                logger.warning(
                    f"Unable to fix {col} because there are not enough training example"
                )
                continue

            if self.__debug_mode:
                logger.warning("Fix under debug mode, use quickest fix algorithm...")
                fixer = get_quick_fixer_for_debug(self.column_types[col])
            else:
                fixer = get_fixer(fix_strategy)

            try:
                fixed_info = fixer.fix(
                    preprocessed_df,
                    errors,
                    self._cached_column_types.get_meta(col),
                )
                if fixed_info:
                    all_fix_info.extend(fixed_info)
                else:
                    self._fix_strategy[col] = FixStrategy.UNABLE_TO_FIX

            except EmptyTrainDataException:
                logger.warning(
                    f"Unable to fix {col} because there are not enough training example"
                )
        return all_fix_info

    def auto_fix(
        self, errors: Optional[CellErrors] = None, *detectors: BaseErrorDetector
    ) -> "DataFrame":
        if errors is None:
            errors = self.detect_error(*detectors)

        preprocessed_df = self._preprocess(errors)

        if self._all_fix_info is None:
            all_fix_info = self._gather_fix_info(errors, preprocessed_df)
            self._all_fix_info = all_fix_info
        else:
            all_fix_info = self._all_fix_info

        fixed_df = preprocessed_df.copy()
        fixed_df = fulfil_fix_back(fixed_df, all_fix_info)
        fixed_df = self._processor.restore(fixed_df)

        refiner = MisplacedRefiner(self._error_detector)
        fixed_df, fix_info_list_for_refine = refiner.refine(
            processed_fixed_df=fixed_df,
        )

        fixed_df = merge_num_with_tag_columns(fixed_df, self.column_types)
        refiner = ValidationRefiner(self._error_detector)
        fixed_df, fix_info_list_for_refine = refiner.refine(
            processed_fixed_df=fixed_df,
        )

        final_fixed_df = finalize_columns(self, fixed_df)

        df = _construct_new_df(
            final_fixed_df,
            self._processor,
            self._error_detector,
            self._fix_strategy,
        )
        df._all_fix_info = all_fix_info
        return df
