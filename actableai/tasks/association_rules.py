import time
from typing import Dict, List
import pandas as pd

from actableai.tasks import TaskType
from actableai.tasks.base import AAITask


class AAIAssociationRulesTask(AAITask):
    @AAITask.run_with_ray_remote(TaskType.ASSOCIATION_RULES)
    def run(
        self,
        df: pd.DataFrame,
        individuals: List[str],
        items: str,
        frequent_method: str = "fpgrowth",
        min_support: float = 0.5,
        association_metric: str = "confidence",
        min_association_metric: float = 0.5,
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

        from actableai.data_validation.checkers import NoFrequentItemSet
        from actableai.data_validation.params import AssociationRulesDataValidator
        from actableai.data_validation.base import CheckLevels

        assert frequent_method in ["fpgrowth", "apriori", "fpmax"]
        assert association_metric in ["confidence", "lift", "leverage", "conviction"]

        start = time.time()
        df = df.copy()

        # Validate parameters
        data_validation_results = AssociationRulesDataValidator().validate(
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
            df_encoded,
            columns=te.columns_,
            index=df_list[individuals].astype(str).apply(", ".join, axis=1),
        )

        # Run the association rule analysis
        frequent_dict = {
            "fpgrowth": fpgrowth,
            "apriori": apriori,
            "fpmax": fpmax,
        }
        frequent_func = frequent_dict[frequent_method]
        frequent_itemset = frequent_func(
            df_encoded,
            min_support=min_support,
            use_colnames=True,
            max_len=None,
            verbose=0,
        )
        check_no_frequent_itemset = NoFrequentItemSet(level=CheckLevels.CRITICAL).check(
            frequent_itemset
        )
        if check_no_frequent_itemset is not None:
            return {
                "status": "FAILURE",
                "data": {},
                "validations": [
                    {
                        "name": check_no_frequent_itemset.name,
                        "level": check_no_frequent_itemset.level,
                        "message": check_no_frequent_itemset.message,
                    }
                ],
                "runtime": time.time() - start,
            }
        rules = association_rules(
            frequent_itemset,
            metric=association_metric,
            min_threshold=min_association_metric,
        )

        return {
            "status": "SUCCESS",
            "data": {"rules": rules},
            "validations": [],
            "runtime": time.time() - start,
        }
