from actableai.data_imputation.meta import ColumnType


POSSIBLE_DATETIME_FORMAT = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%d/%m/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S.%fZ",
)

REGEX_CONSTS = {
    ColumnType.Temperature: r"[+-]?\d+(?:\.\d+)?\s*°\s*[cCfF]",
    ColumnType.Percentage: r"^[+-]?([0-9]*[.])?[0-9]+\s*%$",
    ColumnType.NumWithTag: r"^\s*([+-]?\d+(?:\.\d+)?)\s*([a-zA-Z_]+)$|^\s*([a-zA-Z_]+)\s*:?\s*([+-]?\d+(?:\.\d+)?)$",
    ColumnType.Complex: (
        r"^(?=[iI.\d+-])([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![iI.\d]))?"
        r"([+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)?[iIjJ])?$"
    ),
}
