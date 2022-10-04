import logging
from tempfile import mkdtemp
from typing import Any, Optional, List, Dict, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from scipy.special import logit, expit
from autogluon.tabular import TabularPredictor
from autogluon.features import AutoMLPipelineFeatureGenerator
from econml.dml import LinearDML, NonParamDML

from actableai.causal.predictors import SKLearnMultilabelWrapper, SKLearnTabularWrapper
from actableai.intervention.config import LOGIT_MAX_VALUE, LOGIT_MIN_VALUE
from actableai.utils import get_type_special_no_ag
from actableai.utils.multilabel_predictor import MultilabelPredictor
from actableai.utils.preprocessors.autogluon_preproc import DMLFeaturizer


class AAIInterventionEffectPredictor:
    def __init__(
        self,
        target: str,
        current_intervention_column: str,
        new_intervention_column: Optional[str],
        expected_target: Optional[str] = None,
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
        automl_pipeline_feature_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.target = target
        self.current_intervention_column = current_intervention_column
        self.new_intervention_column = new_intervention_column
        self.expected_target = expected_target
        self.common_causes = common_causes
        self.causal_cv = 1 if causal_cv is None else causal_cv
        self.causal_hyperparameters = causal_hyperparameters
        self.cate_alpha = cate_alpha
        self.presets = presets
        self.model_directory = model_directory
        self.num_gpus = num_gpus
        self.feature_importance = feature_importance
        self.drop_unique = drop_unique
        self.drop_useless_features = drop_useless_features
        self.automl_pipeline_feature_parameters = automl_pipeline_feature_parameters
        self.causal_model = None
        self.outcome_featurizer = None
        if automl_pipeline_feature_parameters is None:
            self.automl_pipeline_feature_parameters = {}

    def _generate_model_t(self, X, T) -> SKLearnTabularWrapper:
        type_special = T.apply(get_type_special_no_ag)
        num_cols = (type_special == "numeric") | (type_special == "integer")
        num_cols = list(T.loc[:, num_cols].columns)
        cat_cols = type_special == "category"
        cat_cols = list(T.loc[:, cat_cols].columns)
        model_t_problem_type = (
            "regression"
            if self.current_intervention_column in num_cols
            else "multiclass"
        )

        model_t_holdout_frac = None
        if model_t_problem_type == "multiclass":
            model_t_holdout_frac = len(
                T[self.current_intervention_column].unique()
            ) / len(T)

        ag_args_fit = {"num_gpus": self.num_gpus, "drop_unique": self.drop_unique}
        feature_generator = AutoMLPipelineFeatureGenerator(
            **(self.automl_pipeline_feature_parameters)
        )
        model_t_predictor = TabularPredictor(
            path=mkdtemp(prefix=str(self.model_directory)),
            label="t",
            problem_type=model_t_problem_type,
        )

        xw_col = []
        if X is not None:
            xw_col += list(X.columns)

        model_t = SKLearnTabularWrapper(
            model_t_predictor,
            x_w_columns=xw_col,
            hyperparameters=self.causal_hyperparameters,
            presets=self.presets,
            ag_args_fit=ag_args_fit,
            feature_generator=feature_generator,
            holdout_frac=model_t_holdout_frac,
        )
        return model_t

    def _generate_model_y(
        self, X, Y
    ) -> Union[SKLearnTabularWrapper, SKLearnMultilabelWrapper]:
        xw_col = []
        if X is not None:
            xw_col += list(X.columns)
        ag_args_fit = {"num_gpus": self.num_gpus, "drop_unique": self.drop_unique}
        feature_generator = AutoMLPipelineFeatureGenerator(
            **(self.automl_pipeline_feature_parameters)
        )
        if len(Y.columns) == 1:
            model_y_predictor = TabularPredictor(
                path=mkdtemp(prefix=str(self.model_directory)),
                label="y",
                problem_type="regression",
            )
            model_y = SKLearnTabularWrapper(
                model_y_predictor,
                x_w_columns=xw_col,
                hyperparameters=self.causal_hyperparameters,
                presets=self.presets,
                ag_args_fit=ag_args_fit,
                feature_generator=feature_generator,
            )
        else:
            model_y_predictor = MultilabelPredictor(
                labels=Y.columns,
                path=mkdtemp(prefix=str(self.model_directory)),
                problem_types=["regression"] * len(Y.columns),
            )
            model_y = SKLearnMultilabelWrapper(
                ag_predictor=model_y_predictor,
                x_w_columns=xw_col,
                hyperparameters=self.causal_hyperparameters,
                presets=self.presets,
                ag_args_fit=ag_args_fit,
                feature_generator=feature_generator,
                holdout_frac=None,
            )
        return model_y

    def _generate_model_final(self, T, Y):
        feature_generator = AutoMLPipelineFeatureGenerator(
            **(self.automl_pipeline_feature_parameters)
        )
        ag_args_fit = {"num_gpus": self.num_gpus, "drop_unique": self.drop_unique}

        if len(Y.columns) == 1:
            # This tabular predictor might be useless we could use only the multilabel
            model_final = TabularPredictor(
                path=mkdtemp(prefix=str(self.model_directory)),
                label="y_res",
                problem_type="regression",
            )
            model_final = SKLearnTabularWrapper(
                model_final,
                x_w_columns=T.columns,
                hyperparameters=self.causal_hyperparameters,
                presets=self.presets,
                ag_args_fit=ag_args_fit,
                feature_generator=feature_generator,
            )
        else:
            model_final_predictor = MultilabelPredictor(
                labels=Y.columns,
                path=mkdtemp(prefix=str(self.model_directory)),
                problem_types=["regression"] * len(Y.columns),
            )
            model_final = SKLearnMultilabelWrapper(
                ag_predictor=model_final_predictor,
                hyperparameters=self.causal_hyperparameters,
                presets=self.presets,
                ag_args_fit=ag_args_fit,
                feature_generator=feature_generator,
                holdout_frac=None,
            )
        return model_final

    def _generate_dml_model(self, model_t, model_y, model_final, X, T):
        T_type = get_type_special_no_ag(T[self.current_intervention_column])
        if (
            self.common_causes is None
            or len(self.common_causes) == 0
            or self.cate_alpha is not None
            or (
                T_type == "category"
                and len(T[self.current_intervention_column].unique()) > 2
            )
        ):
            # Multiclass treatment (One treatment but categorical)
            causal_model = LinearDML(
                model_t=model_t,
                model_y=model_y,
                featurizer=None
                if self.common_causes is None or len(self.common_causes) == 0
                else DMLFeaturizer(),
                cv=self.causal_cv,
                linear_first_stages=False,
                discrete_treatment=T_type == "category",
            )
        else:
            causal_model = NonParamDML(
                model_t=model_t,
                model_y=model_y,
                model_final=model_final,
                featurizer=None if X is None else DMLFeaturizer(),
                cv=self.causal_cv,
                discrete_treatment=T_type == "category",
            )
        return causal_model

    def fit(self, df: pd.DataFrame, target_proba: Optional[pd.DataFrame] = None):
        type_special = df.apply(get_type_special_no_ag)
        num_cols = (type_special == "numeric") | (type_special == "integer")
        num_cols = list(df.loc[:, num_cols].columns)
        cat_cols = type_special == "category"
        cat_cols = list(df.loc[:, cat_cols].columns)

        T0, _, Y, X = self._generate_TYX(df, target_proba, True)

        model_t = self._generate_model_t(X, T0)
        model_y = self._generate_model_y(X, Y)
        model_final = self._generate_model_final(T0, Y)
        self.causal_model = self._generate_dml_model(
            model_t, model_y, model_final, X, T0
        )
        self.causal_model.fit(
            Y=Y.values,
            T=T0.values,
            X=X.values if X is not None else None,
            cache_values=True,
        )

        return self

    def predict_effect(self, df, target_proba):
        result = pd.DataFrame()

        if self.causal_model is None:
            raise NotFittedError()
        T0, T1, Y, X = self._generate_TYX(df, target_proba, False)

        effects = self.causal_model.effect(
            X,
            T0=T0,  # type: ignore
            T1=T1,  # type: ignore
        )

        target_intervened = None
        target_intervened = Y + effects
        if self.outcome_featurizer is not None:
            target_intervened = self.outcome_featurizer.inverse_transform(
                expit(target_intervened)
            )

        result[self.target + "_intervened"] = target_intervened.squeeze()  # type: ignore
        if len(Y.columns) == 1:
            result["intervention_effect"] = effects.squeeze()
            if self.cate_alpha is not None:
                lb, ub = self.causal_model.effect_interval(
                    X,
                    T0=T0,  # type: ignore
                    T1=T1,  # type: ignore
                    alpha=self.cate_alpha,
                )  # type: ignore
                result[self.target + "_intervened_low"] = df[self.target] + lb.flatten()
                result[self.target + "_intervened_high"] = (
                    df[self.target] + ub.flatten()
                )
                result["intervention_effect_low"] = lb.flatten()
                result["intervention_effect_high"] = ub.flatten()
        return result

    def _generate_TYX(
        self, df, target_proba, fit
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        type_special = df.apply(get_type_special_no_ag)
        num_cols = (type_special == "numeric") | (type_special == "integer")
        num_cols = list(df.loc[:, num_cols].columns)
        cat_cols = type_special == "category"
        cat_cols = list(df.loc[:, cat_cols].columns)
        X = (
            df[self.common_causes]
            if self.common_causes and len(self.common_causes) > 0
            else None
        )
        if self.target in num_cols:
            Y = df[[self.target]]
        else:
            if fit:
                self.outcome_featurizer = OneHotEncoder(
                    sparse=False, handle_unknown="ignore"
                )
                self.outcome_featurizer.fit(df[[self.target]])
            if target_proba is not None:
                Y = target_proba
            else:
                Y = pd.DataFrame(
                    self.outcome_featurizer.transform(df[[self.target]]),
                    columns=self.outcome_featurizer.get_feature_names_out(),
                )
            Y = pd.DataFrame(logit(Y)).clip(LOGIT_MIN_VALUE, LOGIT_MAX_VALUE)
        T0 = df[[self.current_intervention_column]]
        T1 = df[[self.new_intervention_column]]
        return T0, T1, Y, X

    def preprocess_data(self, df) -> pd.DataFrame:
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
        return df

    def check_params(self, df, target_proba):
        type_special = df.apply(get_type_special_no_ag)
        num_cols = (type_special == "numeric") | (type_special == "integer")
        num_cols = list(df.loc[:, num_cols].columns)
        if self.target in num_cols and target_proba is not None:
            logging.warning(
                "`df[target]` is a numerical column and `target_proba` is not None: `target_proba` will be ignored"
            )
        if self.target not in num_cols and self.cate_alpha is not None:
            logging.warning(
                "`df[target]` is a categorical column and `cate_alpha` is not None: `cate_alpha` will be ignored"
            )
        return

    def predict_two_way_effect(self, df):
        # Only works for treatment + outcome being continuous
        # Maybe raise Exception if not called well

        if self.causal_model is None:
            raise NotFittedError()

        T0, T1, Y, X = self._generate_TYX(df, None, False)

        new_inter = [None for _ in range(len(df))]
        new_out = [None for _ in range(len(df))]
        ctr = T0
        cta = df[self.expected_target]
        custom_effect = []
        for index_lab in range(len(df)):
            curr_CME = self.causal_model.const_marginal_effect(
                X[index_lab : index_lab + 1] if X is not None else None
            )
            curr_CME = curr_CME.squeeze()
            custom_effect.append(curr_CME)
        CME = np.array(custom_effect)
        # New Outcome
        if self.new_intervention_column in df:
            ntr = df[self.new_intervention_column]
            new_out = (ntr - ctr) * CME + cta
        # New Intervention
        if self.expected_target in df:
            nta = df[self.expected_target]
            new_inter = ((nta - cta) / CME) + ctr
        return pd.DataFrame(
            {
                self.expected_target: new_out,
                self.new_intervention_column: new_inter,
            }
        )
