from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import norm

from actableai.tasks.base import AAITask
from actableai.tasks import TaskType
from actableai.tasks.causal_inference import AAICausalInferenceTask
from actableai.data_validation.params import CausalFeatureSelectionDataValidator
from actableai.data_validation.base import CLASSIFICATION_MINIMUM_NUMBER_OF_CLASS_SAMPLE


class AAIDirectCausalFeatureSelection(AAITask):
    """Search for direct causal features with DML."""

    @AAITask.run_with_ray_remote(TaskType.DIRECT_CAUSAL_FEATURE_SELECTION)
    def run(
        self,
        df,
        target,
        features,
        max_concurrent_ci_tasks=4,
        dummy_prefix_sep=":::",
        positive_outcome_value=None,
        causal_inference_task_params=None,
    ):
        failed_checks = CausalFeatureSelectionDataValidator().validate(
            target, features, df
        )
        validations = [
            {"name": x.name, "level": x.level, "message": x.message}
            for x in failed_checks
        ]
        features = filter(
            lambda c: df[c].dtype != "object"
            or df[c].nunique() <= CLASSIFICATION_MINIMUM_NUMBER_OF_CLASS_SAMPLE,
            features
        )

        dummies = pd.get_dummies(df[features], prefix_sep=dummy_prefix_sep)
        dummy_features = dummies.columns
        dummies[target] = df[target]
        result, future = {}, {}
        with ThreadPoolExecutor(max_workers=max_concurrent_ci_tasks) as executor:
            for feature in dummy_features:
                feature_group = feature.split(dummy_prefix_sep)[0]
                other_features = [
                    f
                    for f in dummy_features
                    if f.split(dummy_prefix_sep)[0] != feature_group
                ]
                future[
                    executor.submit(
                        self._run_ci,
                        dummies,
                        feature,
                        target,
                        other_features,
                        positive_outcome_value,
                        causal_inference_task_params or {},
                    )
                ] = feature

        for f in as_completed(future):
            feature = future[f]
            re = f.result()
            if "status" in re:
                return re
            result[feature] = re
        return {"status": "SUCCESS", "data": result, "validations": validations}

    def _run_ci(
        self,
        dummies,
        feature,
        target,
        common_causes,
        positive_outcome_value,
        causal_inference_task_params,
    ):
        task = AAICausalInferenceTask(**causal_inference_task_params)
        re = task.run(
            dummies,
            [feature],
            [target],
            common_causes=common_causes,
            positive_outcome_value=positive_outcome_value,
        )
        if re["status"] != "SUCCESS":
            return {
                "status": "FAILURE",
                "messenger": re.get("messenger", ""),
                "validations": re["validations"],
            }

        y_res, t_res = re["data"]["Y_res"], re["data"]["T_res"]
        prod = y_res * t_res
        khi = np.mean(prod)
        sigmatwo = np.mean(np.square((prod - khi)))
        pvalue = 2 * (
            1 - norm.cdf(abs(khi), 0, np.sqrt(sigmatwo) / np.sqrt(len(dummies)))
        )
        is_direct_cause = pvalue < 0.1 / (len(dummies.columns) - 1)
        return {
            "effect": re["data"]["effect"],
            "pvalue": pvalue,
            "is_direct_cause": is_direct_cause,
            "khi": khi,
            "sigmatwo": sigmatwo,
        }
