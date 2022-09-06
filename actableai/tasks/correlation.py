import pandas as pd
from typing import List, Optional, Dict

from actableai.tasks import TaskType
from actableai.tasks.base import AAITask


class AAICorrelationTask(AAITask):
    """Correlation Task

    Args:
        AAITask: Base Class for tasks
    """

    @AAITask.run_with_ray_remote(TaskType.CORRELATION)
    def run(
        self,
        df: pd.DataFrame,
        target_column: str,
        target_value: Optional[str] = None,
        kde_steps: int = 100,
        lr_steps: int = 100,
        control_columns: Optional[List[str]] = None,
        control_values: Optional[List[str]] = None,
        correlation_threshold: float = 0.05,
        p_value: float = 0.05,
        use_bonferroni: bool = False,
        top_k: int = 20,
    ) -> Dict:
        """Runs a correlation analysis on Input DataFrame

        Args:
            df: Input DataFrame
            target_column: Target for correlation analysis
            target_value: If target_column type is categorical, target_value must be one
                value of target_column. Else should be None. Defaults to None.
            kde_steps: Number of steps for kernel density graph. Defaults to 100.
            lr_steps: Number of steps for linear regression graph. Defaults to 100.
            control_columns: Control columns for decorrelations. Defaults to None.
            control_values: Control values for decorrelations. control_values[i] must be
                a value from df[control_columns[i]]. Defaults to None.
            correlation_threshold: Threshold for correlation validation. Values with
                an absolute value above this threshold are considered correlated.
                Defaults to 0.05.
            p_value: PValue for correlation validation.
                Defaults to 0.05.
            use_bonferroni (bool, optional): Whether we should use bonferroni test.
                Defaults to False.
            top_k: Limit for number of results returned. Only the best k correlated
                columns are returned. Defaults to 20.

        Examples:
            >>> df = pd.read_csv("path/to/dataframe")
            >>> result = AAICorrelationTask().run(
            ...     df,
            ...     ["feature1", "feature2", "feature3"],
            ...     "target"
            ... )

        Returns:
            Dict: Dictionnary containing the results
                - "status": "SUCCESS" if the task successfully ran else "FAILURE"
                - "messenger": Message returned with the task
                - "data": Dictionary containing the data for the clustering task
                    - "corrs": Correlation values for each feature
                    - "charts": Dictionnary containing the charts for correlations
                - "runtime": Time taken to run the task
                - "validations": List of validations on the data,
                    non-empty if the data presents a problem for the task
        """
        import logging
        import time
        import numpy as np
        from sklearn.neighbors import KernelDensity
        from sklearn.linear_model import BayesianRidge
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from autogluon.features import (
            DatetimeFeatureGenerator,
        )
        from pandas.api.types import (
            is_datetime64_any_dtype,
            is_bool_dtype,
            is_object_dtype,
        )

        from actableai.stats import Stats
        from actableai.data_validation.params import CorrelationDataValidator
        from actableai.utils import is_fitted, is_text_column
        from actableai.data_validation.base import CheckLevels
        from actableai.utils.preprocessors.preprocessing import (
            SKLearnAGFeatureWrapperBase,
            MultiCountVectorizer,
        )
        from actableai.utils.preprocessors.autogluon_preproc import (
            CustomeDateTimeFeatureGenerator,
        )
        from actableai.correlation.config import (
            N_GRAM_RANGE_MIN,
            N_GRAM_RANGE_MAX,
            N_GRAM_MAX_FEATURES,
        )

        if use_bonferroni:
            p_value /= len(df.columns) - 1

        start = time.time()

        # To resolve any issues of acces rights make a copy
        df = df.copy()

        data_validation_results = CorrelationDataValidator().validate(df, target_column)
        failed_checks = [x for x in data_validation_results if x is not None]
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

        if target_value is not None:
            df[target_column] = df[target_column].astype(str)
            target_value = str(target_value)

        # Type reader
        date_cols = df.apply(is_datetime64_any_dtype)

        ct = ColumnTransformer(
            [
                (
                    DatetimeFeatureGenerator.__name__,
                    SKLearnAGFeatureWrapperBase(CustomeDateTimeFeatureGenerator()),
                    date_cols,
                ),
            ],
            remainder="passthrough",
            sparse_threshold=0,
            verbose_feature_names_out=False,
            verbose=True,
        )
        df = pd.DataFrame(
            ct.fit_transform(df).tolist(), columns=ct.get_feature_names_out()
        )

        bool_cols = df.apply(is_bool_dtype)
        df.loc[:, bool_cols] = df.loc[:, bool_cols].astype(str)

        str_cols = df.apply(is_object_dtype)
        cat_cols = str_cols | bool_cols
        df.loc[:, cat_cols] = df.loc[:, cat_cols].fillna("None")
        df.loc[:, cat_cols] = df.loc[:, cat_cols].astype(str)

        text_cols = df.apply(is_text_column) & cat_cols
        cat_cols = cat_cols & ~text_cols

        # Set as categorical if target_value given
        if target_value is not None:
            mask_cols = df.columns == target_column
            cat_cols = cat_cols | mask_cols
            text_cols = text_cols & ~mask_cols

        og_df_col = df.columns
        og_target_col = df.loc[:, cat_cols]

        ct = ColumnTransformer(
            [
                (OneHotEncoder.__name__, OneHotEncoder(), cat_cols),
                (
                    MultiCountVectorizer.__name__,
                    MultiCountVectorizer(
                        ngram_range=(N_GRAM_RANGE_MIN, N_GRAM_RANGE_MAX),
                        max_features=N_GRAM_MAX_FEATURES,
                    ),
                    text_cols,
                ),
            ],
            remainder="passthrough",
            sparse_threshold=0,
            verbose_feature_names_out=False,
            verbose=True,
        )
        df = pd.DataFrame(
            ct.fit_transform(df).tolist(), columns=ct.get_feature_names_out()
        )

        if control_columns is not None or control_values is not None:
            if len(control_columns) != len(control_values):
                return {
                    "status": "FAILURE",
                    "messenger": "control_columns and control_values must have the same length",
                }

            for control_column, control_value in zip(control_columns, control_values):
                try:
                    idx = Stats().decorrelate(
                        df,
                        target_column,
                        control_column,
                        target_value=target_value,
                        control_value=control_value,
                    )
                except Exception:
                    logging.exception("Fail to de-correlate.")
                    return {
                        "status": "FAILURE",
                        "messenger": "Can't de-correlate {} from {}".format(
                            control_column, target_column
                        ),
                    }
                if idx.shape[0] == 0:
                    return {
                        "status": "FAILURE",
                        "messenger": "De-correlation returns empty data",
                    }
                df = df.loc[idx]

        if df.shape[0] < 3:
            return {
                "status": "FAILURE",
                "messenger": "Not enough data to calculate correlation",
                "validations": [],
            }

        cat_cols = []
        gen_cat_cols = []
        if is_fitted(ct.named_transformers_["OneHotEncoder"]):
            cat_cols = og_df_col[ct.transformers[0][2]]
            gen_cat_cols = ct.named_transformers_["OneHotEncoder"].categories_
        corrs = Stats().corr(
            df,
            target_column,
            target_value,
            p_value=p_value,
            categorical_columns=cat_cols,
            gen_categorical_columns=gen_cat_cols,
        )
        corrs = corrs[:top_k]

        df = df.join(og_target_col)
        charts = []
        other = (
            lambda uniques, label: uniques[uniques != label][0]
            if uniques.size == 2
            else "others"
        )

        def kde_bandwidth(x):
            return max(0.5 * x.std() * (x.size ** (-0.2)), 1e-2)

        for corr in corrs:
            if type(corr["col"]) is list:
                group, val = corr["col"]

                df[group].fillna("None", inplace=True)

                # Categorical variable
                if target_value is None:
                    # Target column is continuous
                    X = np.linspace(
                        df[target_column].min(), df[target_column].max(), kde_steps
                    )

                    x1 = df[target_column][df[group] == val]
                    x1 = x1[x1.notna()]
                    k1 = KernelDensity(bandwidth=kde_bandwidth(x1)).fit(
                        x1.values.reshape((-1, 1))
                    )

                    x2 = df[target_column][df[group] != val]
                    x2 = x2[x2.notna()]
                    k2 = KernelDensity(bandwidth=kde_bandwidth(x2)).fit(
                        x2.values.reshape((-1, 1))
                    )

                    charts.append(
                        {
                            "type": "kde",
                            "corr": corr["corr"],
                            "data": [
                                {
                                    "value": val,
                                    "y": np.exp(
                                        k1.score_samples(X.reshape((-1, 1)))
                                    ).tolist(),
                                    "x": X.tolist(),
                                    "group": group,
                                    "y_label": target_column,
                                },
                                {
                                    "value": other(df[group].unique(), val),
                                    "y": np.exp(
                                        k2.score_samples(X.reshape(-1, 1))
                                    ).tolist(),
                                    "x": X.tolist(),
                                    "group": group,
                                    "y_label": target_column,
                                },
                            ],
                        }
                    )
                else:
                    # Target value is also categorical
                    x = df[target_column].copy()
                    x[x != target_value] = other(
                        df[target_column].unique(), target_value
                    )

                    y = df[group].copy()
                    y[y != val] = other(df[group], val)

                    charts.append(
                        {
                            "type": "cm",
                            "corr": corr["corr"],
                            "data": {
                                "cm": pd.crosstab(
                                    x, y, dropna=False, normalize="index"
                                ).to_dict(),
                                "corr": corr,
                            },
                        }
                    )

            else:
                if target_value is None:
                    X, y = df[corr["col"]], df[target_column]
                    idx = X.notna() & y.notna()
                    X, y = X[idx], y[idx]
                    clf = BayesianRidge(compute_score=True)
                    clf.fit(X.values.reshape((-1, 1)), y)
                    r2 = clf.score(X.values.reshape((-1, 1)), y)
                    x_pred = np.linspace(X.min(), X.max(), lr_steps)
                    y_mean, y_std = clf.predict(
                        x_pred.reshape((-1, 1)), return_std=True
                    )
                    charts.append(
                        {
                            "type": "lr",
                            "corr": corr["corr"],
                            "data": {
                                "x": X.tolist(),
                                "y": y.tolist(),
                                "x_pred": x_pred.tolist(),
                                "intercept": clf.intercept_,
                                "coef": clf.coef_[0],
                                "r2": r2,
                                "y_mean": y_mean.tolist(),
                                "y_std": y_std.tolist(),
                                "x_label": corr["col"],
                                "y_label": target_column,
                            },
                        }
                    )
                else:
                    col = corr["col"]
                    X = np.linspace(df[col].min(), df[col].max(), kde_steps)

                    x1 = df[col][df[target_column] == target_value]
                    x1 = x1[x1.notna()]
                    k1 = KernelDensity(bandwidth=kde_bandwidth(x1)).fit(
                        x1.values.reshape((-1, 1))
                    )

                    x2 = df[col][df[target_column] != target_value]
                    x2 = x2[x2.notna()]
                    k2 = KernelDensity(bandwidth=kde_bandwidth(x2)).fit(
                        x2.values.reshape((-1, 1))
                    )

                    charts.append(
                        {
                            "type": "kde",
                            "corr": corr["corr"],
                            "data": [
                                {
                                    "value": target_value,
                                    "y": np.exp(
                                        k1.score_samples(X.reshape((-1, 1)))
                                    ).tolist(),
                                    "x": X.tolist(),
                                    "group": target_column,
                                    "y_label": col,
                                },
                                {
                                    "value": other(
                                        df[target_column].unique(), target_value
                                    ),
                                    "y": np.exp(
                                        k2.score_samples(X.reshape(-1, 1))
                                    ).tolist(),
                                    "x": X.tolist(),
                                    "group": target_column,
                                    "y_label": col,
                                },
                            ],
                        }
                    )

        runtime = time.time() - start

        return {
            "status": "SUCCESS",
            "messenger": "",
            "runtime": runtime,
            "data": {
                "corr": corrs,
                "charts": charts,
            },
            "validations": [
                {"name": x.name, "level": x.level, "message": x.message}
                for x in failed_checks
            ],
        }
