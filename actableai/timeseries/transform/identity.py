from typing import Iterable

from gluonts.dataset import DataEntry

from actableai.timeseries.transform.base import Transformation


class Identity(Transformation):
    """
    TODO write documentation
    """

    def transform(self, data_it: Iterable[DataEntry]) -> Iterable[DataEntry]:
        """
        TODO write documentation
        """
        return data_it
