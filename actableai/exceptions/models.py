from typing import List


class MissingFeaturesException(KeyError):
    """Exception representing missing features when doing model inference"""

    def __init__(self, missing_features: List[str], *args):
        """
        TODO write documentation
        """
        super().__init__(missing_features, *args)
        self.missing_features = missing_features

    def __str__(self):
        return f"The following features are missing: {self.missing_features}"
