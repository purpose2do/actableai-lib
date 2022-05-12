import pandas as pd
from datetime import datetime
from typing import List, Text

from actableai.data_imputation.type_recon.regex_consts import POSSIBLE_DATETIME_FORMAT


def as_datetime(
    series: pd.Series,
    possible_format: List[Text] = POSSIBLE_DATETIME_FORMAT,
) -> pd.Series:
    possible_format = list(possible_format)

    def convert_to_datetime(date_string: Text):
        for i, format_string in enumerate(possible_format):
            try:
                dt = datetime.strptime(date_string, format_string)
                possible_format[0], possible_format[i] = (
                    possible_format[i],
                    possible_format[0],
                )
                return dt
            except ValueError:
                continue
        return date_string

    return series.apply(convert_to_datetime)
