from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.stats import norm

from actableai.tasks.base import AAITask
from actableai.tasks import TaskType
from actableai.tasks.causal_inference import AAICausalInferenceTask
from actableai.data_validation.params import CausalFeatureSelectionDataValidator


class AAIDirectCausalFeatureSelection(AAITask):
    """Search for direct causal features with DML."""

    @AAITask.run_with_ray_remote(TaskType.DIRECT_CAUSAL_FEATURE_SELECTION)
    def run(
        self,
        df,
        target,
        features,
        max_concurrent_ci_tasks=4,
        positive_outcome_value=None,
        causal_inference_task_params=None,
        causal_inference_run_params=None,
    ):
        """
        This function performs causal feature selection on a given dataset.

        Args:
            self (object): The instance of the class.
            df (pandas.DataFrame): The dataframe containing the data.
            target (str): The target feature for which the causal inference is to be performed.
            features (list): List of features for which the causal inference is to be performed.
            max_concurrent_ci_tasks (int, optional): Maximum number of concurrent causal inference tasks. Defaults to 4.
            dummy_prefix_sep (str, optional): Prefix separator to be used while creating dummy variables. Defaults to ":::".
            positive_outcome_value (str, optional): Positive outcome value.
            causal_inference_task_params (dict, optional): Causal inference task parameters. Defaults to None.
            causal_inference_run_params (dict, optional): Causal inference run parameters. Defaults to None.

        Returns:
            dict: A dictionary containing the status, data and validations of the function.

        """
        df = df.copy()

        validation_checks = CausalFeatureSelectionDataValidator().validate(
            target, features, df
        )
        failed_checks = [check for check in validation_checks if check is not None]
        validations = [
            {"name": x.name, "level": x.level, "message": x.message}
            for x in failed_checks
        ]

        str_columns = df[features].select_dtypes(include="object").columns
        for c in str_columns:
            vc = df[c].value_counts()
            median_key = vc[vc <= vc.median()].idxmax()
            df.loc[df[c] != median_key, c] = 0
            df.loc[df[c] == median_key, c] = 1
            df = df.astype({c: "int32"})

        result, future = {}, {}
        with ThreadPoolExecutor(max_workers=max_concurrent_ci_tasks) as executor:
            for feature in features:
                other_features = [f for f in features if f != feature]
                future[
                    executor.submit(
                        self._run_ci,
                        df,
                        feature,
                        target,
                        other_features,
                        positive_outcome_value,
                        causal_inference_task_params or {},
                        causal_inference_run_params or {},
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
        df,
        feature,
        target,
        common_causes,
        positive_outcome_value,
        causal_inference_task_params,
        causal_inference_run_params,
    ):
        """
        This function runs a causal inference task using the AAICausalInferenceTask
        class, and returns the effect, p-value, and whether or not the feature is a
        direct cause of the target.

        Args:
        df (pandas dataframe): Input dataframe.
        feature (str): Name of feature in the dataframe to use as the independent variable.
        target (str): Name of target in the dataframe to use as the dependent variable.
        common_causes (list of str): List of names of variables in the dataframe to use as common causes.
        positive_outcome_value (int or float): Value of the positive outcome in the target variable.
        causal_inference_task_params (dict): Dictionary of parameters to pass to the AAICausalInferenceTask class.
        causal_inference_run_params (dict): Dictionary of parameters to pass to the run() method of AAICausalInferenceTask class.

        Returns:
        A dictionary containing the following fields:
        'effect': The estimated causal effect of the feature on the target.
        'pvalue': The p-value for the estimated causal effect.
        'is_direct_cause': A Boolean indicating whether or not the feature is a direct cause of the target.
        'khi': mean of the product of y_res and t_res
        'sigmatwo': mean of the square of the difference between the product of y_res and t_res and khi

        """
        task = AAICausalInferenceTask(**causal_inference_task_params)
        re = task.run(
            df,
            [feature],
            [target],
            common_causes=common_causes,
            positive_outcome_value=positive_outcome_value,
            **causal_inference_run_params,
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
        pvalue = 2 * (1 - norm.cdf(abs(khi), 0, np.sqrt(sigmatwo) / np.sqrt(len(df))))
        is_direct_cause = pvalue < 0.1 / (len(df.columns) - 1)
        return {
            "effect": re["data"]["effect"],
            "pvalue": pvalue,
            "is_direct_cause": is_direct_cause,
            "khi": khi,
            "sigmatwo": sigmatwo,
        }
