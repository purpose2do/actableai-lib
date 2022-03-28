from actableai.data_imputation.config import logger
from actableai.data_imputation.error_detector.base_error_detector import (
    BaseErrorDetector,
)
from actableai.data_imputation.error_detector.cell_erros import (
    CellErrors,
    CellError,
    ErrorType,
)
from actableai.data_imputation.meta.column import NumWithTagColumnMeta


class NullDetector(BaseErrorDetector):
    def detect_cells(self) -> CellErrors:
        errors = CellErrors(self._dftypes)

        for col in self._df.columns:
            logger.info(f"Detecting NULL errors for {col}")
            new_errors = self._df[col][self._df[col].isna()]
            col_meta = self._dftypes.get_meta(col)
            if isinstance(col_meta, NumWithTagColumnMeta):
                for index in new_errors.index:
                    errors.append(
                        CellError(col_meta.get_num_column_name(), index, ErrorType.NULL)
                    )
                    errors.append(
                        CellError(
                            col_meta.get_left_tag_column_name(), index, ErrorType.NULL
                        )
                    )
                    errors.append(
                        CellError(
                            col_meta.get_right_tag_column_name(), index, ErrorType.NULL
                        )
                    )
            else:
                for index in new_errors.index:
                    errors.append(CellError(col, index, ErrorType.NULL))

        return errors
