from io import StringIO
import time
from typing import List, Dict, Optional
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from actableai.data_validation.base import CheckLevels

from actableai.tasks import TaskType
from actableai.tasks.base import AAITask
from actableai.utils import memory_efficient_hyperparameters


class AAIInterventionTask(AAITask):
    @AAITask.run_with_ray_remote(TaskType.INTERVENTION)
    def run(
        self,
        df: pd.DataFrame,
        target: str,
        current_intervention_column: str,
        new_intervention_column: str,
        common_causes: Optional[List[str]] = None,
        causal_cv: Optional[int] = None,
        causal_hyperparameters: Optional[Dict] = None,
        cate_alpha: Optional[float] = None,
        presets: Optional[str] = None,
        model_directory: Optional[str] = None,
        num_gpus: Optional[int] = 0,
        feature_importance: Optional[bool] = True,
        drop_unique: bool = True,
        drop_useless_features: bool = True,
    ) -> Dict:
        """Run this intervention task and return the results.

        Args:
            df: Input DataFrame
            target: Column name of target variable
            current_intervention_column: Column name of the current intervention
            new_intervention_column: Column name of the new intervention
            common_causes: List of common causes to be used for the intervention
            causal_cv: Number of folds for causal cross validation
            causal_hyperparameters: Hyperparameters for AutoGluon
                See https://auto.gluon.ai/stable/api/autogluon.task.html?highlight=tabularpredictor#autogluon.tabular.TabularPredictor
            cate_alpha: Alpha for intervention effect
            presets: Presets for AutoGluon.
                See https://auto.gluon.ai/stable/api/autogluon.task.html?highlight=tabularpredictor#autogluon.tabular.TabularPredictor
            model_directory: Model directory
            num_gpus: Number of GPUs used by causal models
            drop_unique: Whether the classification algorithm drops columns that
                only have a unique value accross all rows at fit time
            drop_useless_features: Whether the classification algorithm drops columns that
                only have a unique value accross all rows at preprocessing time

        Examples:
            >>> import pandas as pd
            >>> from actableai import AAIInterventionTask
            >>> df = pd.read_csv("path/to/csv")
            >>> result = AAIInterventionTask().run(
            ...     df,
            ...     'target_column',
            ... )

        Returns:
            Dict: Dictionnay containing the following keys:
                - status: Status of the task
                - messenger: Message of the task
                - validations: Validations for the tasks parameters
                - data: Dictionnary containing the following keys:
                    - df: DataFrame with the intervention
                    - causal_graph_dot: Causal graph in dot format
                    - T_res: Residuals of the treatment
                    - Y_res: Residuals of the outcome
                    - X: Common causes
                    - model_t_scores: Model scores for the treatment
                    - model_y_scores: Model scores for the outcome
                    - intervention_plot: Data for plotting the intervention
                - runtime: Runtime of the task
        """
        from tempfile import mkdtemp
        from econml.dml import LinearDML, NonParamDML
        from autogluon.tabular import TabularPredictor
        from autogluon.features.generators import AutoMLPipelineFeatureGenerator
        from dowhy import CausalModel
        import numpy as np
        import networkx as nx

        from actableai.data_validation.params import InterventionDataValidator
        from actableai.causal.predictors import SKLearnWrapper
        from actableai.utils.preprocessors.autogluon_preproc import DMLFeaturizer
        from actableai.utils import get_type_special_no_ag

        start = time.time()
        # Handle default parameters
        if model_directory is None:
            model_directory = mkdtemp(prefix="autogluon_model")
        if common_causes is None:
            common_causes = []
        if len(common_causes) == 0:
            drop_unique = False
        if presets is None:
            presets = "medium_quality_faster_train"
        if causal_hyperparameters is None:
            causal_hyperparameters = memory_efficient_hyperparameters()
        causal_cv = 1 if causal_cv is None else causal_cv

        automl_pipeline_feature_parameters = {}
        if not drop_useless_features:
            automl_pipeline_feature_parameters["pre_drop_useless"] = False
            automl_pipeline_feature_parameters["post_generators"] = []

        df = df.copy()

        # Validate parameters
        data_validation_results = InterventionDataValidator().validate(
            df,
            target,
            current_intervention_column,
            new_intervention_column,
            common_causes,
            causal_cv,
            drop_unique,
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

        # Preprocess data
        type_special = df.apply(get_type_special_no_ag)
        num_cols = (type_special == "numeric") | (type_special == "integer")
        num_cols = list(df.loc[:, num_cols].columns)
        cat_cols = type_special == "category"
        cat_cols = list(df.loc[:, cat_cols].columns)
        df = df.replace(to_replace=[None], value=np.nan)
        if len(num_cols):
            df.loc[:, num_cols] = SimpleImputer(strategy="median").fit_transform(
                df.loc[:, num_cols]
            )
        if len(cat_cols):
            df.loc[:, cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(
                df.loc[:, cat_cols]
            )

        X = df[common_causes] if len(common_causes) > 0 else None

        model_t_problem_type = (
            "regression" if current_intervention_column in num_cols else "multiclass"
        )

        model_t_holdout_frac = None
        if model_t_problem_type == "multiclass" and len(df) > 0:
            model_t_holdout_frac = len(df[current_intervention_column].unique()) / len(
                df
            )

        model_t = TabularPredictor(
            path=mkdtemp(prefix=str(model_directory)),
            label="t",
            problem_type=model_t_problem_type,
        )

        xw_col = []
        if X is not None:
            xw_col += list(X.columns)

        model_t = SKLearnWrapper(
            model_t,
            x_w_columns=xw_col,
            hyperparameters=causal_hyperparameters,
            presets=presets,
            ag_args_fit={"num_gpus": num_gpus, "drop_unique": drop_unique},
            feature_generator=AutoMLPipelineFeatureGenerator(
                **automl_pipeline_feature_parameters
            ),
            holdout_frac=model_t_holdout_frac,
        )

        model_y = TabularPredictor(
            path=mkdtemp(prefix=str(model_directory)),
            label="y",
            problem_type="regression",
        )
        model_y = SKLearnWrapper(
            model_y,
            x_w_columns=xw_col,
            hyperparameters=causal_hyperparameters,
            presets=presets,
            ag_args_fit={"num_gpus": num_gpus, "drop_unique": drop_unique},
            feature_generator=AutoMLPipelineFeatureGenerator(
                **automl_pipeline_feature_parameters
            ),
        )

        if (
            X is None
            or cate_alpha is not None
            or (
                current_intervention_column in cat_cols
                and len(df[current_intervention_column].unique()) > 2
            )
        ):
            # Multiclass treatment
            causal_model = LinearDML(
                model_t=model_t,
                model_y=model_y,
                featurizer=None if X is None else DMLFeaturizer(),
                cv=causal_cv,
                linear_first_stages=False,
                discrete_treatment=current_intervention_column in cat_cols,
            )
        else:
            model_final = TabularPredictor(
                path=mkdtemp(prefix=str(model_directory)),
                label="y_res",
                problem_type="regression",
            )
            model_final = SKLearnWrapper(
                model_final,
                hyperparameters=causal_hyperparameters,
                presets=presets,
                ag_args_fit={
                    "num_gpus": num_gpus,
                    "drop_unique": drop_unique,
                },
                feature_generator=AutoMLPipelineFeatureGenerator(
                    **automl_pipeline_feature_parameters
                ),
            )
            causal_model = NonParamDML(
                model_t=model_t,
                model_y=model_y,
                model_final=model_final,
                featurizer=None if X is None else DMLFeaturizer(),
                cv=causal_cv,
                discrete_treatment=current_intervention_column in cat_cols,
            )

        causal_model.fit(
            df[[target]].values,
            df[[current_intervention_column]].values,
            X=X.values if X is not None else None,
            cache_values=True,
        )

        effects = causal_model.effect(
            X.values if X is not None else None,
            T0=df[[current_intervention_column]],  # type: ignore
            T1=df[[new_intervention_column]],  # type: ignore
        )

        df[target + "_intervened"] = df[target] + effects.flatten()  # type: ignore
        df["intervention_effect"] = effects.flatten()  # type: ignore
        if cate_alpha is not None:
            lb, ub = causal_model.effect_interval(
                X.values if X is not None else None,
                T0=df[[current_intervention_column]],  # type: ignore
                T1=df[[new_intervention_column]],  # type: ignore
                alpha=cate_alpha,
            )  # type: ignore
            df[target + "_intervened_low"] = df[target] + lb.flatten()
            df[target + "_intervened_high"] = df[target] + ub.flatten()
            df["intervention_effect_low"] = lb.flatten()
            df["intervention_effect_high"] = ub.flatten()

        # Construct Causal Graph
        buffer = StringIO()
        causal_model_do_why = CausalModel(
            data=df,
            treatment=[current_intervention_column],
            outcome=target,
            common_causes=common_causes,
        )
        nx.drawing.nx_pydot.write_dot(causal_model_do_why._graph._graph, buffer)  # type: ignore # noqa
        causal_graph_dot = buffer.getvalue()

        Y_res, T_res, X_, W_ = causal_model.residuals_

        # This has lots of redundant code with causal_inferences.py
        # Should be refactored
        estimation_results = {
            "causal_graph_dot": causal_graph_dot,
            "T_res": T_res,
            "Y_res": Y_res,
            "X": X_,
            "df": df,
        }

        model_t_scores, model_y_scores = [], []
        for i in range(len(causal_model.nuisance_scores_t)):
            model_t_scores.append(
                r2_score(
                    np.concatenate([n["y"] for n in causal_model.nuisance_scores_t[i]]),
                    np.concatenate(
                        [n["y_pred"] for n in causal_model.nuisance_scores_t[i]]
                    ),
                )
            )
            model_y_scores.append(
                r2_score(
                    np.concatenate([n["y"] for n in causal_model.nuisance_scores_y[i]]),
                    np.concatenate(
                        [n["y_pred"] for n in causal_model.nuisance_scores_y[i]]
                    ),
                )
            )

        estimation_results["model_t_scores"] = {
            "values": model_t_scores,
            "mean": np.mean(model_t_scores),
            "stderr": np.std(model_t_scores) / np.sqrt(len(model_t_scores)),
            "metric": "r2",
        }

        estimation_results["model_y_scores"] = {
            "values": model_y_scores,
            "mean": np.mean(model_y_scores),
            "stderr": np.std(model_y_scores) / np.sqrt(len(model_y_scores)),
            "metric": "r2",
        }

        if feature_importance and X is not None:
            importances = []
            # Only run feature importance for first mc_iter to speed it up
            for _, m in enumerate(causal_model.models_t[0]):
                importances.append(m.feature_importance())
            model_t_feature_importances = sum(importances) / causal_cv
            model_t_feature_importances["stderr"] = model_t_feature_importances[
                "stddev"
            ] / np.sqrt(causal_cv)
            model_t_feature_importances.sort_values(
                ["importance"], ascending=False, inplace=True
            )
            estimation_results[
                "model_t_feature_importances"
            ] = model_t_feature_importances

            importances = []
            for _, m in enumerate(causal_model.models_y[0]):
                importances.append(m.feature_importance())
            model_y_feature_importances = sum(importances) / causal_cv
            model_y_feature_importances["stderr"] = model_y_feature_importances[
                "stddev"
            ] / np.sqrt(causal_cv)
            model_y_feature_importances.sort_values(
                ["importance"], ascending=False, inplace=True
            )
            estimation_results[
                "model_y_feature_importances"
            ] = model_y_feature_importances

        # Display plot in front end
        intervention_names = None
        intervention_diff = None
        pair_dict = None
        if current_intervention_column in num_cols:
            intervention_diff = (
                df[new_intervention_column] - df[current_intervention_column]
            )
        else:
            intervention_names = (
                df[current_intervention_column] + " -> " + df[new_intervention_column]
            )
            pair_dict = {}
            for _, val in df.iterrows():
                intervention_name = (
                    val[current_intervention_column]
                    + " -> "
                    + val[new_intervention_column]
                )
                if intervention_name not in pair_dict:
                    pair_dict[intervention_name] = {
                        "original_target": [],
                        "target_intervened": [],
                        "intervention_effect": [],
                    }
                pair_dict[intervention_name]["original_target"].append(val[target])
                pair_dict[intervention_name]["target_intervened"].append(
                    val[target + "_intervened"]
                )
                pair_dict[intervention_name]["intervention_effect"].append(
                    val["intervention_effect"]
                )
        estimation_results["intervention_plot"] = {
            "type": "category"
            if current_intervention_column in cat_cols
            else "numeric",
            "intervention_diff": intervention_diff,
            "intervention_names": intervention_names,
            "min_target": df[target].min(),
            "max_target": df[target].max(),
            "pair_dict": pair_dict,
        }

        return {
            "status": "SUCCESS",
            "messenger": "",
            "validations": [
                {"name": x.name, "level": x.level, "message": x.message}
                for x in failed_checks
            ],
            "data": estimation_results,
            "runtime": time.time() - start,
        }
