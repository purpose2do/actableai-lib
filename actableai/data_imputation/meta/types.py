from enum import Enum


class ColumnType(Enum):
    Id = "id"
    Category = "category"
    String = "string"
    Text = "text"
    Integer = "integer"
    Float = "float"
    Complex = "complex"
    Timestamp = "timestamp"
    Temperature = "temperature"
    Percentage = "percentage"
    NumWithTag = "num_with_tag"
    Unknown = "unknown"
    NULL = "null"


ColumnTypeUnsupported = {
    ColumnType.Id,
    ColumnType.Text,
    ColumnType.String,
    ColumnType.Temperature,
    ColumnType.NULL,
}
