from typing import Iterable

from gluonts.dataset import DataEntry

from actableai.timeseries.transform.base import Transformation


class Identity(Transformation):
    """Identity transformation (no transformation)."""

    def transform(self, data_it: Iterable[DataEntry]) -> Iterable[DataEntry]:
        """Transform data entries.

        Args:
            data_it: Iterable object of data entries.

        Returns:
            Iterable object of transformed data entries.
        """
        return data_it
