from typing import List, Dict, Optional
import pandas as pd

from actableai.tasks import TaskType
from actableai.tasks.base import AAITask


class AAIInterventionTask(AAITask):
    @AAITask.run_with_ray_remote(TaskType.CAUSAL_INFERENCE)
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
    ) -> Dict:
        """Runs an intervention on Input DataFrame

        Args:
            df: Input DataFrame
            target: Target column name
            current_intervention_column: Feature before intervention
            new_intervention_column: Feature after intervention
            common_causes: Common causes
            causal_cv: Cross-validation folds for causal inference
            causal_hyperparameters: Hyperparameters for causal inference
            cate_alpha: CATE alpha
            presets: Presets for causal inference
            model_directory: Model directory
            num_gpus: Number of GPUs to use by the causal model

        Returns:
            Dict: Dictionnay containing the following keys:
                - 'df': DataFrame with the intervention
        """
        import pandas as pd
        from tempfile import mkdtemp
        from econml.dml import LinearDML, NonParamDML
        from sklearn.impute import SimpleImputer
        from autogluon.tabular import TabularPredictor

        from actableai.causal.predictors import SKLearnWrapper
        from actableai.causal import OneHotEncodingTransformer
        from actableai.utils import debiasing_hyperparameters
        from actableai.utils import memory_efficient_hyperparameters

        # Handle default parameters
        if model_directory is None:
            model_directory = mkdtemp(prefix="autogluon_model")
        if causal_hyperparameters is None:
            causal_hyperparameters = debiasing_hyperparameters()
        if common_causes is None:
            common_causes = []
        if presets is None:
            presets = "medium_quality_faster_train"

        df_ = df.copy()
        df_ = df_[pd.notnull(df_[[current_intervention_column, target]]).all(axis=1)]

        df_imputed = df_.copy()
        num_cols = df_imputed.select_dtypes(include="number").columns
        cat_cols = df_imputed.select_dtypes(exclude="number").columns
        if len(num_cols) > 0:
            df_imputed[num_cols] = SimpleImputer(strategy="median").fit_transform(
                df_imputed[num_cols]
            )
        if len(cat_cols) > 0:
            df_imputed[cat_cols] = SimpleImputer(
                strategy="most_frequent"
            ).fit_transform(df_imputed[cat_cols])

        X = df_imputed[common_causes] if len(common_causes) > 0 else None
        model_t = TabularPredictor(
            path=mkdtemp(prefix=str(model_directory)),
            label="t",
            problem_type="regression"
            if current_intervention_column in num_cols
            else "multiclass",
        )
        model_t = SKLearnWrapper(
            model_t,
            hyperparameters=causal_hyperparameters,
            presets=presets,
            ag_args_fit={
                "num_gpus": num_gpus,
            },
        )

        model_y = TabularPredictor(
            path=mkdtemp(prefix=str(model_directory)),
            label="y",
            problem_type="regression",
        )
        model_y = SKLearnWrapper(
            model_y,
            hyperparameters=causal_hyperparameters,
            presets=presets,
            ag_args_fit={
                "num_gpus": num_gpus,
            },
        )

        if (
            (X is None)
            or (cate_alpha is not None)
            or (
                current_intervention_column in cat_cols
                and len(df[current_intervention_column].unique()) > 2
            )
        ):
            # Multiclass treatment
            causal_model = LinearDML(
                model_t=model_t,
                model_y=model_y,
                featurizer=None if X is None else OneHotEncodingTransformer(X),
                cv=causal_cv,
                linear_first_stages=False,
                discrete_treatment=current_intervention_column not in num_cols,
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
                },
            )
            causal_model = NonParamDML(
                model_t=model_t,
                model_y=model_y,
                model_final=model_final,
                featurizer=None if X is None else OneHotEncodingTransformer(X),
                cv=causal_cv,
                discrete_treatment=current_intervention_column not in num_cols,
            )

        causal_model.fit(
            df_[[target]].values,
            df_[[current_intervention_column]].values,
            X=X,
        )

        df_intervene = df_[
            pd.notnull(df_[[current_intervention_column, new_intervention_column]]).all(
                axis=1
            )
        ]
        df_imputed = df_imputed.loc[df_intervene.index]
        X = df_imputed[common_causes] if len(common_causes) > 0 else None
        effects = causal_model.effect(
            X,
            T0=df_imputed[[current_intervention_column]],
            T1=df_imputed[[new_intervention_column]],
        )

        targets = df_intervene[target].copy()
        df_intervene[target + "_intervened"] = targets + effects.flatten()
        if cate_alpha is not None:
            lb, ub = causal_model.effect_interval(
                X,
                T0=df_intervene[[current_intervention_column]],
                T1=df_intervene[[new_intervention_column]],
                alpha=cate_alpha,
            )
            df_intervene[target + "_intervened_low"] = targets + lb.flatten()
            df_intervene[target + "_intervened_high"] = targets + ub.flatten()

        df_intervene["intervention_effect"] = effects.flatten()
        if cate_alpha is not None:
            df_intervene["intervention_effect_low"] = lb.flatten()
            df_intervene["intervention_effect_high"] = ub.flatten()

        return df_intervene
