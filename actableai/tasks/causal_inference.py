import pandas as pd
from pathlib import Path
from typing import List, Optional, Union
from sklearn.preprocessing import LabelEncoder

from actableai.utils import get_type_special
from actableai.tasks import TaskType
from actableai.tasks.base import AAITask


class LogCategoricalOutcomeNotAllowed(ValueError):
    pass


class LogCategoricalTreatmentNotAllowed(ValueError):
    pass


# fit the model with data
def convert_categorical_to_numeric(df, columns):
    transformers = {}
    for c in columns:
        if get_type_special(df[c]) == "category":
            s = df[c].astype(str)
            le = LabelEncoder()
            le.fit(s)
            df[c] = le.transform(s)
            transformers[c] = le

    return df[columns], transformers


class AAICausalInferenceTask(AAITask):
    @AAITask.run_with_ray_remote(TaskType.CAUSAL_INFERENCE)
    def run(
        self,
        pd_table: pd.DataFrame,
        treatments: List,
        outcomes: List,
        effect_modifiers: Optional[List] = None,
        common_causes: Optional[List] = None,
        instrumental_variables: Optional[List] = None,
        controls: Optional[dict] = None,
        positive_outcome_value=None,
        target_units: Optional[str] = "ate",
        alpha: Optional[float] = 0.05,
        tree_max_depth: Optional[int] = 3,
        log_treatment: Optional[bool] = False,
        log_outcome: Optional[bool] = False,
        model_directory: Optional[Union[str, Path]] = None,
        ag_hyperparameters: Optional[Union[str, dict]] = "auto",
        ag_presets: str = "medium_quality_faster_train",
        model_params: Optional[List] = None,
        rscorer: Optional[List] = None,
        cv: Union[int, str] = "auto",
        feature_importance: bool = False,
        mc_iters="auto",
        seed=123,
        num_gpus=0,
        drop_unique: bool = True,
        drop_useless_features: bool = False,
    ):
        """Causal analysis task

        Args:
            pd_table: Dataset for the causal analysis
            treatments: treatment variable(s)
            outcomes: outcome variables
            effect_modifiers: list of effect modifiers (X) for CATE estimation. Defaults to [].
            common_causes: list of common causes (W). Defaults to [].
            instrumental_variables: list of instrumental variables (Z). Defaults to [].
            controls: dictionary of control treatment values. Keys are categorical treatment names
            positive_outcome_value: If not None, target is converted into 0, 1 where 1 is when original target is equal to positive_outcome_value else 0.
            target_units: Targeted used for calculating the effect. Possible values are "ate", "att", "atc". Defaults to "ate"
            alpha: Significance level of effect confidence interval (from 0.01 to 0.99). Defaults to 0.05
            tree_max_depth: Maximum depth of CATE function's tree interpreter. Default to 3.
            log_treatment: flag to indicate whether log transform is to be applied to treatment
            log_outcome: flag to indicate whether log transform is to be applied to outcome
            model_directory: Where the model should be stored, if None stored in the /tmp folder
            ag_hyperparameters: Dictionary of hyperparameters for Autogluon predictors if used
            ag_presets: Presets for Autogluon Models
            model_params: List of model Parameters for the AAICausalEstimator
            rscorer: tune rscorer object
            cv: Number of cross validation for LinearDML
            feature_importance: Whether the feature importance are computed
            mc_iters: Random State for LinearDML. See
                    https://econml.azurewebsites.net/_autosummary/econml.dml.LinearDML.html#econml.dml.LinearDML
            seed: Random numpy seed
            num_gpus: Number of GPUs for the TabularPredictors. Can be set to an int or "auto"
            drop_unique: Wether to drop columns with only unique values as preprocessing step
            drop_useless_features: Whether to drop columns with only unique values at fit time

        Examples:
            >>> df = pd.read_csv("path/to/csv")
            >>> result = infer_causal(
            ...    df,
            ...    treatments=["feature1", "feature2"],
            ...    outcomes=["feature3", "feature4"],
            ...    effect_modifiers=["feature5", "feature6"]
            ... )
            >>> result

        Returns:
            Dict: dictionary of estimation results
                - "status": "SUCCESS" if the task successfully ran else "FAILURE"
                - "messenger": Custom message from the task
                - "runtime": Execution time from the task
                - "data": Dictionnary containing data from the task
                    - "":
                    - "effect":
                    - "controls": Transformed controls
                    - "causal_graph_dot": Feature graph in dot format
                    - "tree_interpreter_dot": Tree interpreter in dot format
                    - "refutation_results": # TODO
                    - "T_res": Treatment residuals
                    - "Y_res": Outcome residuals
                    - "X": Common causes values
                    - "model_t_scores": Scores for the treatment model
                    - "model_y_scores": Scores for the outcome model
                    - "model_t_feature_importances": Feature importances for treatment model
                    - "model_y_feature_importances": Feature importances for outcome model
                - "validations": List of validations on the data,
                        non-empty if the data presents a problem for the task
        """
        import time
        from io import StringIO

        import numpy as np
        import networkx as nx
        from tempfile import mkdtemp
        from actableai.causal.models import AAICausalEstimator
        from actableai.causal.params import (
            DeepIVParams,
        )
        from actableai.causal.tree_utils import make_pretty_tree
        from actableai.causal import has_categorical_column
        from actableai.causal import prepare_sanitize_data

        from actableai.data_validation.base import CheckLevels

        from actableai.data_validation.params import CausalDataValidator
        from dowhy import CausalModel
        from econml.cate_interpreter import SingleTreeCateInterpreter
        from econml.iv.nnet import DeepIV
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import r2_score

        np.random.seed(seed)

        if model_directory is None:
            model_directory = mkdtemp(prefix="autogluon_model")

        start = time.time()
        if effect_modifiers is None:
            effect_modifiers = []
        if common_causes is None:
            common_causes = []
        if instrumental_variables is None:
            instrumental_variables = []
        if controls is None:
            controls = {}
        columns = effect_modifiers + common_causes

        # if controls is provided, convert numeric to categorical
        for c in controls:
            if c in pd_table.columns:
                pd_table = pd_table.astype({c: str})
                controls[c] = str(controls[c])

        data_validation_results = CausalDataValidator().validate(
            treatments,
            outcomes,
            pd_table,
            effect_modifiers,
            common_causes,
            positive_outcome_value,
            drop_unique,
            cv,
        )
        failed_checks = [x for x in data_validation_results if x is not None]

        if CheckLevels.CRITICAL in [x.level for x in failed_checks]:
            return {
                "status": "FAILURE",
                "validations": [
                    {"name": x.name, "level": x.level, "message": x.message}
                    for x in failed_checks
                ],
                "runtime": time.time() - start,
                "data": {},
            }

        pd_table = prepare_sanitize_data(
            pd_table, treatments, outcomes, effect_modifiers, common_causes
        )

        is_single_treatment = len(treatments) == 1
        is_single_outcome = len(outcomes) == 1
        if positive_outcome_value is not None:
            assert (
                len(outcomes) == 1
            ), "Only one outcome is allowed when positive_outcome_value is not None"
        has_categorical_treatment = has_categorical_column(pd_table, treatments)
        has_categorical_outcome = has_categorical_column(pd_table, outcomes)

        if log_treatment:
            if has_categorical_treatment:
                raise LogCategoricalTreatmentNotAllowed()
            for c in treatments:
                pd_table = pd_table[pd_table[c] > 0]
                pd_table[c] = np.log(pd_table[c])
        if log_outcome:
            if has_categorical_outcome:
                raise LogCategoricalOutcomeNotAllowed()
            for c in outcomes:
                pd_table = pd_table[pd_table[c] > 0]
                pd_table[c] = np.log(pd_table[c])
        pd_table = pd_table.reset_index()

        is_single_binary_treatment = False
        if is_single_treatment and has_categorical_treatment:
            if len(pd_table[treatments[0]].unique()) == 2:
                is_single_binary_treatment = True
        is_single_binary_outcome = False
        if is_single_outcome and (
            len(pd_table[outcomes[0]].unique()) == 2
            or positive_outcome_value is not None
        ):
            is_single_binary_outcome = True
            pd_table[outcomes[0]] = pd_table[outcomes[0]].astype(str)
            positive_outcome_value = str(positive_outcome_value)

        # construct the dictionary of control values for categorical treatments
        for c in treatments:
            if get_type_special(pd_table[c]) == "category":
                if c not in controls:
                    treatment_values = sorted(pd_table[c].unique())
                    controls[c] = treatment_values[0]
        non_controls = {}
        for c in controls:
            non_controls[c] = sorted(
                set(pd_table[c].unique()).difference([controls[c]])
            )

        # convert boolean to int
        for c in pd_table.columns:
            if get_type_special(pd_table[c]) == "boolean":
                pd_table[c] = pd_table[c].astype(int)

        # construct the causal graph dot string
        causal_model = CausalModel(
            data=pd_table,
            treatment=treatments,
            outcome=outcomes,
            common_causes=effect_modifiers + common_causes,
            instruments=instrumental_variables,
        )
        buffer = StringIO()
        nx.drawing.nx_pydot.write_dot(causal_model._graph._graph, buffer)
        causal_graph_dot = buffer.getvalue()

        # initalize model_params depending on treatment and outcome numbers and types
        label_t = treatments[0] if is_single_treatment else treatments
        label_y = outcomes[0] if is_single_outcome else outcomes

        if instrumental_variables:
            model_params += [
                DeepIVParams(
                    num_instruments=len(instrumental_variables),
                    num_effect_modifiers=len(effect_modifiers),
                    num_treatments=len(treatments),
                )
            ]

        if positive_outcome_value is not None:
            pd_table[outcomes[0]] = (
                pd_table[outcomes[0]] == positive_outcome_value
            ).astype(int)
        Y = pd_table[outcomes]

        pd_table[treatments], tm_transformers = convert_categorical_to_numeric(
            pd_table, treatments
        )
        T = pd_table[treatments]
        # convert target control and treatment dictionaries to numeric
        for c in controls:
            controls[c] = tm_transformers[c].transform([controls[c]])[0]
        for c in non_controls:
            non_controls[c] = tm_transformers[c].transform(non_controls[c])

        X = None
        W = None
        Z = None

        if effect_modifiers:
            # convert str to numeric if any
            (
                pd_table[effect_modifiers],
                em_transformers,
            ) = convert_categorical_to_numeric(pd_table, effect_modifiers)
            X = pd_table[effect_modifiers]
        if common_causes:
            W = pd_table[common_causes]
        if instrumental_variables:
            pd_table[instrumental_variables], _ = convert_categorical_to_numeric(
                pd_table, instrumental_variables
            )
            Z = pd_table[instrumental_variables]

        ce = AAICausalEstimator(
            model_params=model_params,
            scorer=rscorer,
            has_categorical_treatment=has_categorical_treatment,
            has_binary_outcome=is_single_binary_outcome,
        )
        ce.fit(
            Y,
            T,
            X=X,
            W=W,
            Z=Z,
            label_t=label_t,
            label_y=label_y,
            target_units=target_units,
            cv=cv,
            feature_importance=feature_importance,
            model_directory=model_directory,
            presets=ag_presets,
            hyperparameters=ag_hyperparameters,
            mc_iters=mc_iters,
            num_gpus=num_gpus,
            drop_unique=drop_unique,
            drop_useless_features=drop_useless_features,
        )
        X_test = None
        target_idx = pd.Series(np.full(len(pd_table), True, dtype=bool))
        if (target_units == "att") and is_single_binary_treatment:
            target_idx = (pd_table[treatments] == 1).all(axis="columns")
        elif (target_units == "atc") and is_single_binary_treatment:
            target_idx = (pd_table[treatments] == 0).all(axis="columns")

        tree_interpreter_dot = ""
        if effect_modifiers:
            X_test_df = (
                pd_table.loc[target_idx, effect_modifiers]
                .drop_duplicates()
                .sort_values(by=effect_modifiers)
            )
            X_test = X_test_df.values
            for c, le in em_transformers.items():
                X_test_df[c] = le.inverse_transform(X_test_df[c])

            if controls:
                effect, lb, ub = {}, {}, {}
                for treatment, T0 in controls.items():
                    for T1 in non_controls[treatment]:
                        k = (
                            treatment,
                            tm_transformers[treatment].inverse_transform([T1])[0],
                        )
                        effect[k], lb[k], ub[k] = ce.effect(
                            X_test, T0=T0, T1=T1, alpha=alpha
                        )
            else:
                effect, lb, ub = ce.effect(X_test, alpha=alpha)

            # exclude DeepIV in shap_values and tree interpreter since it doesn't seem to support
            if type(ce.estimator) not in [DeepIV]:
                # construct tree interpreter dot string
                intrp = SingleTreeCateInterpreter(
                    include_model_uncertainty=True, max_depth=tree_max_depth
                )
                intrp.interpret(ce.estimator, X)
                tree_interpreter_dot = intrp.export_graphviz(
                    feature_names=effect_modifiers
                )
                for cat_name in em_transformers:
                    unique_vals = sorted(pd_table[cat_name].unique())
                    cat_vals = {
                        v: em_transformers[cat_name].inverse_transform([v])[0]
                        for v in unique_vals
                    }
                    tree_interpreter_dot = make_pretty_tree(
                        tree_interpreter_dot, [cat_name], [cat_vals]
                    )
        else:
            if controls:
                effect, lb, ub = {}, {}, {}
                for treatment, T0 in controls.items():
                    for T1 in non_controls[treatment]:
                        k = (
                            treatment,
                            tm_transformers[treatment].inverse_transform([T1])[0],
                        )
                        effect_arr, lb_arr, ub_arr = ce.effect(
                            X_test, T0=T0, T1=T1, alpha=alpha
                        )
                        effect[k] = effect_arr[0][0]
                        lb[k] = lb_arr[0][0]
                        ub[k] = ub_arr[0][0]
            else:
                effect, lb, ub = ce.effect(alpha=alpha)

        # parse effect to list of records
        if effect_modifiers:
            if type(effect) is dict:
                effect_records = []
                for i in range(len(effect)):
                    effect_records += X_test_df.to_dict("records")
                start_idx = 0
                for (treatment, T1), v in effect.items():
                    for i in range(len(v)):
                        effect_records[start_idx + i]["treatment_name"] = treatment
                        effect_records[start_idx + i]["treatment_value"] = T1
                        effect_records[start_idx + i]["cate"] = v[i][0]
                        effect_records[start_idx + i]["lb"] = lb[(treatment, T1)][i][0]
                        effect_records[start_idx + i]["ub"] = ub[(treatment, T1)][i][0]
                    start_idx += len(v)
            else:
                effect_records = X_test_df.to_dict("records")
                for i in range(len(effect)):
                    effect_records[i]["cate"] = effect[i][0]
                    effect_records[i]["lb"] = lb[i][0]
                    effect_records[i]["ub"] = ub[i][0]
        else:
            if controls:
                effect_records = []
                for (treatment, T1), v in effect.items():
                    effect_records.append(
                        {
                            "treatment_name": treatment,
                            "treatment_value": T1,
                            "cate": v,
                            "lb": lb[(treatment, T1)],
                            "ub": ub[(treatment, T1)],
                        }
                    )
            else:
                effect_records = [
                    {"cate": effect[0][0], "lb": lb[0][0], "ub": ub[0][0]}
                ]

        # TODO add refutation here

        Y_res, T_res, X_, W_ = ce.estimator.residuals_
        estimation_results = {
            "effect": effect_records,
            "controls": controls,
            "causal_graph_dot": causal_graph_dot,
            "tree_interpreter_dot": tree_interpreter_dot,
            "refutation_results": {},
            "T_res": T_res,
            "Y_res": Y_res,
            "X": X_,
        }

        if len(columns) > 0:
            model_t_scores, model_y_scores = [], []
            for i in range(len(ce.estimator.nuisance_scores_t)):
                model_t_scores.append(
                    (accuracy_score if has_categorical_treatment else r2_score)(
                        np.concatenate(
                            [n["y"] for n in ce.estimator.nuisance_scores_t[i]]
                        ),
                        np.concatenate(
                            [n["y_pred"] for n in ce.estimator.nuisance_scores_t[i]]
                        ),
                    )
                )
                model_y_scores.append(
                    (accuracy_score if is_single_binary_outcome else r2_score)(
                        np.concatenate(
                            [n["y"] for n in ce.estimator.nuisance_scores_y[i]]
                        ),
                        np.concatenate(
                            [n["y_pred"] for n in ce.estimator.nuisance_scores_y[i]]
                        ),
                    )
                )

            estimation_results["model_t_scores"] = {
                "values": model_t_scores,
                "mean": np.mean(model_t_scores),
                "stderr": np.std(model_t_scores) / np.sqrt(len(model_t_scores)),
                "metric": "accuracy" if has_categorical_treatment else "r2",
            }

            estimation_results["model_y_scores"] = {
                "values": model_y_scores,
                "mean": np.mean(model_y_scores),
                "stderr": np.std(model_y_scores) / np.sqrt(len(model_y_scores)),
                "metric": "accuracy" if is_single_binary_outcome else "r2",
            }
            if feature_importance:
                estimation_results[
                    "model_t_feature_importances"
                ] = ce.model_t_feature_importances
                estimation_results[
                    "model_y_feature_importances"
                ] = ce.model_y_feature_importances

        runtime = time.time() - start
        return {
            "status": "SUCCESS",
            "messenger": "",
            "runtime": runtime,
            "data": estimation_results,
            "validations": [],
        }
