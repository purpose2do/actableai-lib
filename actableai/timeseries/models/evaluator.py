from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd
from gluonts.evaluation import Evaluator, MultivariateEvaluator


class AAITimeSeriesEvaluator:
    """Custom Wrapper around GluonTS Evaluator."""

    def __init__(
        self,
        *,
        target_columns: List[str],
        group_list: List[Tuple[Any, ...]] = None,
        **kwargs,
    ):
        """AAITimeSeriesEvaluator Constructor.

        Args:
            target_columns: List of columns to forecast.
            group_list: List of groups, if None will consider only one group called
                ("group",).
            args: The arguments to pass to the GluonTS Evaluator.
            kwargs: The named arguments to pass to the GluonTS Evaluator.
        """
        self.target_columns = target_columns

        self.group_list = group_list

        if len(self.target_columns) <= 1:
            evaluator_class = Evaluator
        else:
            evaluator_class = MultivariateEvaluator

        rmse = lambda target, forecast: np.sqrt(np.mean(np.square(target - forecast)))

        if "custom_eval_fn" not in kwargs:
            kwargs["custom_eval_fn"] = {}
        kwargs["custom_eval_fn"]["_custom_RMSE"] = [rmse, "mean", "median"]

        self.evaluator = evaluator_class(**kwargs)

    def __call__(
        self, *args, **kwargs
    ) -> Tuple[Dict[Tuple[Any, ...], pd.DataFrame], pd.DataFrame]:
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

        # Post-process metrics
        # item_metrics
        if self.group_list is not None:
            target_list = []
            for target in self.target_columns:
                target_list += [target] * len(self.group_list)
            df_item_metrics["target"] = target_list
            df_item_metrics["group"] = self.group_list * len(self.target_columns)
        df_item_metrics = df_item_metrics.reset_index(drop=True)
        df_item_metrics = df_item_metrics.rename(columns={"_custom_RMSE": "RMSE"})

        # agg_metrics
        if len(self.target_columns) <= 1:
            df_agg_metrics = pd.DataFrame(
                [{"target": self.target_columns[0], **agg_metrics}]
            )
        else:
            metric_list = list(agg_metrics.keys())[
                (len(agg_metrics) // (len(self.target_columns) + 1))
                * len(self.target_columns) :
            ]
            df_agg_metrics = pd.DataFrame(columns=["target"] + metric_list)

            for target_index, target_column in enumerate(self.target_columns):
                target_agg_metrics = {
                    metric: agg_metrics[f"{target_index}_{metric}"]
                    for metric in metric_list
                }
                df_agg_metrics = pd.concat(
                    [
                        df_agg_metrics,
                        pd.DataFrame([{"target": target_column, **target_agg_metrics}]),
                    ],
                    ignore_index=True,
                )

        df_agg_metrics = df_agg_metrics.drop(columns="RMSE").rename(
            columns={"_custom_RMSE": "RMSE"}
        )

        df_item_metrics_dict = {}
        if self.group_list is not None:
            for group, df_group in df_item_metrics.groupby("group"):
                df_item_metrics_dict[group] = df_group
        else:
            df_item_metrics_dict[("group",)] = df_item_metrics_dict

        return df_item_metrics_dict, df_agg_metrics
