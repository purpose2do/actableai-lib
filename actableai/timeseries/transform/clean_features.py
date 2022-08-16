from typing import Tuple, Any

from gluonts.dataset import DataEntry
from gluonts.dataset.field_names import FieldName

from actableai.timeseries.transform.base import MapTransformation


class CleanFeatures(MapTransformation):
    """Transformation that clean features from dataset."""

    def __init__(
        self,
        keep_feat_static_real: bool = True,
        keep_feat_static_cat: bool = True,
        keep_feat_dynamic_real: bool = True,
        keep_feat_dynamic_cat: bool = True,
    ):
        """CleanFeatures transformation constructor.

        Args:
            keep_feat_static_real: If False the real static features will be filtered
                out.
            keep_feat_static_cat: If False the categorical static features will be
                filtered out.
            keep_feat_dynamic_real: If False the real dynamic features will be filtered
                out.
            keep_feat_dynamic_cat: If False the categorical dynamic features will be
                filtered out.
        """
        super().__init__()

        self.keep_feat_static_real = keep_feat_static_real
        self.keep_feat_static_cat = keep_feat_static_cat
        self.keep_feat_dynamic_real = keep_feat_dynamic_real
        self.keep_feat_dynamic_cat = keep_feat_dynamic_cat

    def map_transform(self, data: DataEntry, group: Tuple[Any, ...]) -> DataEntry:
        """Transform a data entry.

        Args:
            data: Data entry to revert.
            group: Data entry's group.

        Returns:
            The transformed data entry.
        """
        if not self.keep_feat_static_real:
            data.pop(FieldName.FEAT_STATIC_REAL, None)
        if not self.keep_feat_static_cat:
            data.pop(FieldName.FEAT_STATIC_CAT, None)
        if not self.keep_feat_dynamic_real:
            data.pop(FieldName.FEAT_DYNAMIC_REAL, None)
        if not self.keep_feat_dynamic_cat:
            data.pop(FieldName.FEAT_DYNAMIC_CAT, None)

        return data
