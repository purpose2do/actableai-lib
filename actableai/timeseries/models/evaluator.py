import numpy as np
import pandas as pd
from typing import Dict, Tuple

from gluonts.evaluation import Evaluator, MultivariateEvaluator


class AAITimeSeriesEvaluator:
    """Custom Wrapper around GluonTS Evaluator."""

    def __init__(self, n_targets: int, *args, **kwargs):
        """AAITimeSeriesEvaluator Constructor.

        Args:
            n_targets: Number of targets.
            args: The arguments to pass to the GluonTS Evaluator.
            kwargs: The named arguments to pass to the GluonTS Evaluator.
        """
        if n_targets <= 1:
            evaluator_class = Evaluator
        else:
            evaluator_class = MultivariateEvaluator

        self.evaluator = evaluator_class(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Run the evaluator.

        Args:
            args: The arguments to pass to the evaluator.
            kwargs: The named arguments to pass to the evaluator.
        """
        agg_metrics, df_item_metrics = self.evaluator(*args, **kwargs)

        if agg_metrics["abs_target_sum"] == 0:
            agg_metrics["abs_target_sum"] += 1e-15  # Add epsilon
            for quantile in self.evaluator.quantiles:
                agg_metrics[quantile.weighted_loss_name] = (
                    agg_metrics[quantile.loss_name] / agg_metrics["abs_target_sum"]
                )

            agg_metrics["mean_wQuantileLoss"] = np.array(
                [
                    agg_metrics[quantile.weighted_loss_name]
                    for quantile in self.evaluator.quantiles
                ]
            ).mean()

        return agg_metrics, df_item_metrics
