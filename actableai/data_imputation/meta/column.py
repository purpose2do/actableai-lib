from typing import Text, Union
from typing_extensions import Literal

from actableai.data_imputation.meta.types import ColumnType

_TAG_WITH_NUM_EXPEND_COLUMN_FORMAT = {
    "NUM_COLUMN": "__{original_name}_num__",
    "L_TAG_COLUMN": "__{original_name}_ltag__",
    "R_TAG_COLUMN": "__{original_name}_rtag__",
}


ColumnName = Text

NumType = Union[Literal["int"], Literal["float"]]


class RichColumnMeta:
    def __init__(self, name: ColumnName, col_type: ColumnType):
        self._name = name
        self._type = col_type

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    def __eq__(self, other):
        if not isinstance(other, RichColumnMeta):
            return False
        return self._name == other._name and self._type == other._type

    def __str__(self):
        return f"{self._name}: {self._type.value}"

    def __repr__(self):
        return self.__str__()


class SingleValueColumnMeta(RichColumnMeta):
    pass


class NumWithTagColumnMeta(RichColumnMeta):
    def __init__(
        self,
        name: ColumnName,
        col_type: ColumnType,
    ):
        super(NumWithTagColumnMeta, self).__init__(name, ColumnType.Float)
        self._original_type = col_type
        self._num_type: NumType = "float"

    @property
    def original_type(self):
        return self._original_type

    def set_num_type(self, num_type: NumType):
        self._num_type = num_type

    @property
    def num_type(self):
        return self._num_type

    def get_num_column_name(self):
        return _TAG_WITH_NUM_EXPEND_COLUMN_FORMAT["NUM_COLUMN"].format(
            original_name=self._name
        )

    def get_left_tag_column_name(self):
        return _TAG_WITH_NUM_EXPEND_COLUMN_FORMAT["L_TAG_COLUMN"].format(
            original_name=self._name
        )

    def get_right_tag_column_name(self):
        return _TAG_WITH_NUM_EXPEND_COLUMN_FORMAT["R_TAG_COLUMN"].format(
            original_name=self._name
        )
