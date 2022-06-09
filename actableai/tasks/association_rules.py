from io import StringIO
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
        group_by: List[str],
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
        import networkx as nx

        from actableai.data_validation.checkers import NoFrequentItemSet
        from actableai.data_validation.params import AssociationRulesDataValidator
        from actableai.data_validation.base import CheckLevels

        # Check parameters
        if frequent_method is None:
            frequent_method = "fpgrowth"
        if min_support is None:
            min_support = 0.5
        if association_metric is None:
            association_metric = "confidence"
        if min_association_metric is None:
            min_association_metric = 0.5
        assert frequent_method in ["fpgrowth", "apriori", "fpmax"]
        assert association_metric in [
            "confidence",
            "lift",
            "leverage",
            "conviction",
        ]

        start = time.time()
        df = df.copy()

        # Validate parameters
        data_validation_results = AssociationRulesDataValidator().validate(
            df, group_by, items
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

        # Data Imputation
        df = df.fillna("None")

        # Encode the data
        df_list = df.groupby(group_by)[items].apply(list).reset_index()
        te = TransactionEncoder()
        df_encoded = te.fit_transform(df_list[items])
        df_encoded = pd.DataFrame(
            df_encoded,
            columns=te.columns_,
            index=df_list[group_by].astype(str).apply(", ".join, axis=1),
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
            verbose=1,
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
        try:
            rules = association_rules(
                frequent_itemset,
                metric=association_metric,
                min_threshold=min_association_metric,
            )
        except KeyError:
            association_metric="support"
            rules = association_rules(
                frequent_itemset,
                metric="support",
                min_threshold=0,
                support_only=True,
            )

        temp_df = rules.copy()
        temp_df["antecedents"] = temp_df["antecedents"].apply(list).astype(str)
        temp_df["consequents"] = temp_df["consequents"].apply(list).astype(str)
        temp_df["weight"] = (temp_df["confidence"] - temp_df["confidence"].min()) / (
            temp_df["confidence"].max() - temp_df["confidence"].min()
        )
        temp_df["penwidth"] = temp_df["weight"]
        temp_df["arrowsize"] = temp_df["weight"] + 0.1
        buffer = StringIO()
        graph = nx.from_pandas_edgelist(
            temp_df,
            source="antecedents",
            target="consequents",
            edge_attr=["penwidth", "weight"],
            create_using=nx.DiGraph(),
        )
        nx.drawing.nx_pydot.write_dot(graph, buffer)  # type: ignore

        return {
            "status": "SUCCESS",
            "data": {
                "rules": rules,
                "frequent_itemset": frequent_itemset,
                "df_list": df_list,
                "graph": buffer.getvalue(),
                "association_metric": association_metric
            },
            "validations": [],
            "runtime": time.time() - start,
        }
