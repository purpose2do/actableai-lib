import time
from typing import Dict
import pandas as pd

from actableai.tasks import TaskType
from actableai.tasks.base import AAITask


class AAIAssociationRuleTask(AAITask):
    @AAITask.run_with_ray_remote(TaskType.ASSOCIATION_RULE)
    def run(
        self,
        df: pd.DataFrame,
        individuals: str,
        items: str,
        frequent_method: str = "fpgrowth",
        min_support: float = 0.01,
        metric: str = "confidence",
    ) -> Dict:
        """A task to run an association rule analysis on the data.

        Args:


        Returns:
            Dictionnary of results
        """
        from mlxtend.preprocessing import TransactionEncoder
        from mlxtend.frequent_patterns import (
            apriori,
            association_rules,
            fpgrowth,
            fpmax,
        )

        from actableai.data_validation.params import AssociationRuleDataValidator
        from actableai.data_validation.base import CheckLevels

        assert frequent_method in ["fpgrowth", "apriori", "fpmax"]
        assert metric in ["confidence", "lift", "leverage", "conviction"]

        start = time.time()
        df = df.copy()

        # Validate parameters
        data_validation_results = AssociationRuleDataValidator().validate(
            df, individuals, items
        )
        failed_checks = [
            check for check in data_validation_results if check is not None
        ]

        if CheckLevels.CRITICAL in [x.level for x in failed_checks]:
            return {
                "status": "FAILURE",
                "data": {},
                "validations": [
                    {"name": check.name, "level": check.level, "message": check.message}
                    for check in failed_checks
                ],
                "runtime": time.time() - start,
            }

        # Encode the data
        df_list = df.groupby(individuals)[items].apply(list).reset_index()
        te = TransactionEncoder()
        df_encoded = te.fit_transform(df_list[items])
        df_encoded = pd.DataFrame(
            df_encoded, columns=te.columns_, index=df_list[individuals]
        )

        # Run the association rule analysis
        frequent_dict = {
            "fpgrowth": fpgrowth,
            "apriori": apriori,
            "fpmax": fpmax,
        }
        frequent_func = frequent_dict[frequent_method]
        frequent_itemsets = frequent_func(
            df_encoded,
            min_support=min_support,
            use_colnames=True,
            max_len=None,
            verbose=0,
        )
        rules = association_rules(frequent_itemsets, metric=metric, min_threshold=0.1)

        return rules
