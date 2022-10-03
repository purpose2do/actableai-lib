from typing import List


class MissingFeaturesError(KeyError):
    """Exception representing missing features when doing model inference."""

    def __init__(self, missing_features: List[str], *args):
        """
        TODO write documentation
        """
        super().__init__(missing_features, *args)
        self.missing_features = missing_features

    def __str__(self):
        return f"The following features are missing: {self.missing_features}"


class InvalidTaskIdError(KeyError):
    """Exception representing invalid task id when doing model inference."""

    pass


class InvalidPositiveLabelError(KeyError):
    """Exception representing a positive label which does not exist."""

    pass

class UnknownModelClassError(Exception):
    """Exception representing a Model being loaded corresponding to no known Model"""

    pass
