from copy import deepcopy

import pandas as pd

from typing import cast, List, Iterator, Tuple, Union, Dict, Optional, Callable

from gluonts.dataset import DataEntry
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.pandas import is_series, series_to_dataframe
from sklearn.preprocessing import LabelEncoder

from actableai.timeseries.utils import (
    handle_datetime_column,
    find_freq,
    find_gluonts_freq,
)
from actableai.utils import get_type_special


class AAITimeSeriesDataset:
    """GluonTS-compatible custom class to store a time series dataset."""

    @staticmethod
    def _split_feature_columns(
        df: pd.DataFrame, feature_columns: List[str], group_by: List[str]
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Split features columns into four groups, static, dynamic, real, and categorical.

        Args:
            df: DataFrame containing the data.
            feature_columns: List of feature columns.
            group_by: List of columns to use to separate different time series/groups.

        Returns:
            - List of columns containing real static features.
            - List of columns containing categorical static features.
            - List of columns containing real dynamic features.
            - List of columns containing categorical dynamic features.
        """
        real_feature_columns = []
        cat_feature_columns = []
        for column in feature_columns:
            if get_type_special(df[column]) == "category":
                cat_feature_columns.append(column)
            else:
                real_feature_columns.append(column)

        # Encode categorical columns
        if len(cat_feature_columns) > 0:
            df[cat_feature_columns] = df[cat_feature_columns].apply(
                LabelEncoder().fit_transform
            )

        df_unique = None
        if len(group_by) > 0:
            df_unique = df.groupby(group_by).nunique().max()
        else:
            df_unique = df.nunique()

        feat_static_real = []
        feat_static_cat = []
        feat_dynamic_real = []
        feat_dynamic_cat = []

        # Real columns
        for column in real_feature_columns:
            if df_unique[column] == 1:
                feat_static_real.append(column)
            else:
                feat_dynamic_real.append(column)

        # Categorical columns
        for column in cat_feature_columns:
            if df_unique[column] == 1:
                feat_static_cat.append(column)
            else:
                feat_dynamic_cat.append(column)

        return (
            feat_static_real,
            feat_static_cat,
            feat_dynamic_real,
            feat_dynamic_cat,
        )

    def __init__(
        self,
        dataframes: Union[
            pd.DataFrame,
            pd.Series,
            List[pd.DataFrame],
            Dict[str, pd.DataFrame],
        ],
        target_columns: List[str],
        freq: Optional[str] = None,
        prediction_length: int = 0,
        training: bool = True,
        group_by: Optional[List[str]] = None,
        date_column: str = None,
        feature_columns: Optional[Union[str, List[str]]] = None,
        feat_dynamic_real: Optional[List[str]] = None,
        feat_dynamic_cat: Optional[List[str]] = None,
        feat_static_real: Optional[List[str]] = None,
        feat_static_cat: Optional[List[str]] = None,
    ):
        """AAITimeSeriesDataset Constructor.

        Args:
            dataframes: Single DataFrame/Series or a collection as list or dict
                containing at least `target_columns` values.
            target_columns: List of columns to forecast.
            freq: The frequency of the time series, if None it will be inferred from the
                data.
            prediction_length: Length of the prediction to forecast.
            training: If False the target data will be trimmed if needed.
            group_by: List of columns to use to separate different time series/groups.
                This list is used by the `groupby` function of the pandas library.
            date_column: Column containing the date/datetime/time component of the time
                series. If None the index of the DataFrame will be used.
            feature_columns: List of columns containing extraneous features used to
                forecast. If one or more feature columns contain dynamic features
                (features that change over time) the dataset must contain
                `prediction_length` features data points in the future.
            feat_dynamic_real: List of dynamic real feature columns.
            feat_dynamic_cat: List of dynamic categorical feature columns.
            feat_static_real: List of static real feature columns.
            feat_static_cat: List of static categorical feature columns.
        """
        self.dataframes = {}
        self.freq = freq
        self.gluonts_freq = None
        self.prediction_length = prediction_length
        self.training = training

        self.target_columns = target_columns
        self.feat_dynamic_real = feat_dynamic_real
        self.feat_dynamic_cat = feat_dynamic_cat
        self.feat_static_real = feat_static_real
        self.feat_static_cat = feat_static_cat

        if not isinstance(self.target_columns, list):
            self.target_columns = [self.target_columns]

        if self.feat_dynamic_real is None:
            self.feat_dynamic_real = []
        if self.feat_dynamic_cat is None:
            self.feat_dynamic_cat = []
        if self.feat_static_real is None:
            self.feat_static_real = []
        if self.feat_static_cat is None:
            self.feat_static_cat = []

        if group_by is None:
            group_by = []

        dataframes = deepcopy(dataframes)

        if is_series(dataframes):
            self.dataframes = series_to_dataframe(dataframes)

        if isinstance(dataframes, dict):
            self.dataframes = dataframes
        elif isinstance(dataframes, list):
            self.dataframes = {
                ("data", str(group_index)): df
                for group_index, df in enumerate(dataframes)
            }
        else:
            if feature_columns is not None:
                (
                    _feat_static_real,
                    _feat_static_cat,
                    _feat_dynamic_real,
                    _feat_dynamic_cat,
                ) = self._split_feature_columns(dataframes, feature_columns, group_by)

                self.feat_dynamic_real += _feat_dynamic_real
                self.feat_dynamic_cat += _feat_dynamic_cat
                self.feat_static_real += _feat_static_real
                self.feat_static_cat += _feat_static_cat

            if len(group_by) > 0:
                for group_index, (group, grouped_df) in enumerate(
                    dataframes.groupby(group_by)
                ):
                    if len(group_by) == 1:
                        group = (group,)

                    self.dataframes[group] = grouped_df.reset_index(drop=True)
            else:
                self.dataframes[("data",)] = dataframes

        self.has_dynamic_features = (
            len(self.feat_dynamic_real) + len(self.feat_dynamic_cat)
        ) > 0

        for group in self.dataframes.keys():
            if date_column is not None:
                self.dataframes[group].index = self.dataframes[group][date_column]
                self.dataframes[group].name = date_column

            pd_date, _ = handle_datetime_column(
                self.dataframes[group].index.to_series()
            )

            # Try to guess the freq
            if self.freq is None:
                self.freq = find_freq(pd_date)

            if self.gluonts_freq is None:
                self.gluonts_freq = find_gluonts_freq(pd_date, self.freq)

            self.dataframes[group].index = pd_date
            self.dataframes[group].sort_index(inplace=True)

            self.dataframes[group] = self.dataframes[group][
                self.target_columns
                + self.feat_dynamic_real
                + self.feat_dynamic_cat
                + self.feat_static_real
                + self.feat_static_cat
            ]

        self.process = ProcessDataEntry(
            cast(str, self.gluonts_freq), one_dim_target=(len(self.target_columns) == 1)
        )

    def _dataentry(self, df: pd.DataFrame) -> DataEntry:
        """Return the data as a DataEntry using the dataset information.

        Args:
            df: The data to convert.

        Returns:
            The Data Entry.
        """
        return self._as_dataentry(data=df)

    def __iter__(self) -> Iterator[DataEntry]:
        """Iterate over the time series, one item per group.

        Returns:
            Iterator of Data Entries.
        """
        for group in self.dataframes.keys():
            dataentry = self.process(self._dataentry(self.dataframes[group]))
            if self.has_dynamic_features and not self.training:
                dataentry = self._prepare_prediction_data(dataentry)

            yield dataentry

    def __len__(self) -> int:
        """Return the number of groups in the dataset.

        Returns:
            Number of groups.
        """
        return len(self.dataframes)

    def slice_data(
        self, slice_df: Optional[Union[slice, Callable]], copy: bool = False
    ) -> "AAITimeSeriesDataset":
        """Slice dataset.

        Args:
            slice_df: Slice or function to call that will return a slice. The slice will be
                applied to each group separately.
            copy: If True the data will be copied.

        Returns:
            The sliced dataset.
        """

        df_dict = {}
        for group, df in self.dataframes.items():
            new_df = df
            if copy:
                new_df = new_df.copy()

            slice_ = slice_df
            if callable(slice_df):
                slice_ = slice_df(df)
            df_dict[group] = new_df.iloc[slice_]

        return AAITimeSeriesDataset(
            dataframes=df_dict,
            target_columns=self.target_columns,
            freq=self.freq,
            prediction_length=self.prediction_length,
            training=self.training,
            feat_dynamic_real=self.feat_dynamic_real,
            feat_dynamic_cat=self.feat_dynamic_cat,
            feat_static_real=self.feat_static_real,
            feat_static_cat=self.feat_static_cat,
        )

    def clean_features(
        self,
        keep_feat_static_real: bool,
        keep_feat_static_cat: bool,
        keep_feat_dynamic_real: bool,
        keep_feat_dynamic_cat: bool,
    ) -> "AAITimeSeriesDataset":
        """Filter out features from dataset.

        Args:
            keep_feat_static_real: Whether to keep the real static features or not.
            keep_feat_static_cat: Whether to keep the categorical static features or
                not.
            keep_feat_dynamic_real: Whether to keep to the real dynamic features or not.
            keep_feat_dynamic_cat: Whether to keep the categorical dynamic features or
                not.

        Returns:
            Filtered Dataset.
        """
        return AAITimeSeriesDataset(
            dataframes=self.dataframes,
            target_columns=self.target_columns,
            freq=self.freq,
            prediction_length=self.prediction_length,
            training=self.training,
            feat_static_real=self.feat_static_real if keep_feat_static_real else None,
            feat_static_cat=self.feat_static_cat if keep_feat_static_cat else None,
            feat_dynamic_real=self.feat_dynamic_real
            if keep_feat_dynamic_real
            else None,
            feat_dynamic_cat=self.feat_dynamic_cat if keep_feat_dynamic_cat else None,
        )

    def _as_dataentry(
        self,
        data: pd.DataFrame,
    ) -> DataEntry:
        """Convert a single time series (uni- or multi-variate) that is given in a
            pandas.DataFrame format to a DataEntry.

        Args:
        data: pandas.DataFrame containing at least `target_columns`.

        Returns:
            The Data Entry.
        """
        start = data.index[0]
        dataentry = {FieldName.START: start}

        def set_field(fieldname, col_names, f=lambda x: x):
            if len(col_names) > 0:
                dataentry[fieldname] = [f(data.loc[:, n].to_list()) for n in col_names]

        if len(self.target_columns) == 1:
            dataentry[FieldName.TARGET] = data.loc[:, self.target_columns[0]].to_list()
        else:
            set_field(FieldName.TARGET, self.target_columns)

        set_field(FieldName.FEAT_DYNAMIC_REAL, self.feat_dynamic_real)
        set_field(FieldName.FEAT_DYNAMIC_CAT, self.feat_dynamic_cat)
        set_field(FieldName.FEAT_STATIC_REAL, self.feat_static_real, lambda x: x[0])
        set_field(FieldName.FEAT_STATIC_CAT, self.feat_static_cat, lambda x: x[0])

        return dataentry

    def _prepare_prediction_data(self, dataentry: DataEntry) -> DataEntry:
        """Remove `prediction_length` values from `target` and
            `past_feat_dynamic_real`.

        Args:
            dataentry: The Data Entry to modify.

        Returns:
            The new modified Data Entry.
        """
        entry = deepcopy(dataentry)
        for fname in [FieldName.TARGET, FieldName.PAST_FEAT_DYNAMIC_REAL]:
            if fname in entry:
                entry[fname] = entry[fname][..., : -self.prediction_length]
        return entry
