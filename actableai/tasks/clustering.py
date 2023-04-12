import pandas as pd
from typing import Dict, List, Optional, Any

from actableai.parameters.options import OptionsParameter
from actableai.parameters.parameters import Parameters
from actableai.tasks import TaskType
from actableai.tasks.base import AAITask


class AAIClusteringTask(AAITask):
    """Clustering Task

    Args:
        AAITask: Base class for every tasks
    """

    @staticmethod
    def _get_clustering_model_parameters() -> OptionsParameter[Parameters]:
        from actableai.clustering.models.base import Model
        from actableai.clustering.models import model_parameters_dict

        available_models = [
            Model.affinity_propagation,
            Model.agglomerative_clustering,
            Model.dbscan,
            Model.dec,
            Model.kmeans,
            Model.spectral_clustering,
        ]

        default_model = Model.dec

        return OptionsParameter[Parameters](
            name="clustering_model",
            display_name="Clustering Model",
            description="Model used for clustering",
            default=default_model,
            is_multi=False,
            options={
                model: {
                    "display_name": model_parameters_dict[model].display_name,
                    "value": model_parameters_dict[model],
                }
                for model in available_models
            },
        )

    @staticmethod
    def _get_projection_model_parameters() -> OptionsParameter[Parameters]:
        from actableai.embedding.models.base import Model
        from actableai.embedding.models import model_parameters_dict

        available_models = [
            Model.linear_discriminant_analysis,
            Model.tsne,
            Model.umap,
        ]

        default_model = Model.linear_discriminant_analysis

        return OptionsParameter[Parameters](
            name="projection_model",
            display_name="Projection Model",
            description="Model used for projection",
            default=default_model,
            is_multi=False,
            options={
                model: {
                    "display_name": model_parameters_dict[model].display_name,
                    "value": model_parameters_dict[model],
                }
                for model in available_models
            },
        )

    @classmethod
    def get_parameters(cls) -> Parameters:
        parameters = [
            cls._get_clustering_model_parameters(),
            cls._get_projection_model_parameters(),
        ]

        return Parameters(
            name="clustering_parameters",
            display_name="Clustering Parameters",
            parameters=parameters,
        )

    @AAITask.run_with_ray_remote(TaskType.CLUSTERING)
    def run(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        num_clusters: int = 2,
        drop_low_info: bool = False,
        explain_samples: bool = False,
        cluster_explain_max_depth=20,
        cluster_explain_min_impurity_decrease=0.001,
        cluster_explain_min_samples_leaf=0.001,
        cluster_explain_min_precision=0.8,
        max_train_samples: Optional[int] = None,
        parameters: Dict[str, Any] = None,
    ) -> Dict:
        """Runs a clustering analysis on df

        Args:
            df: Input DataFrame
            features: Features used in Input DataFrame. Defaults to None.
            num_clusters: Number of different clusters assignable to each row.
            drop_low_info: Wether the algorithm drops columns with only one unique
                value or only different categorical values accross all rows.
            explain_samples: If the result contains a human readable explanation of
                the clustering.
            init: Initialization for weights of the DEC model.
            pretrain_optimizer: Optimizer for pretaining phase of autoencoder.
            update_interval: The interval to check the stopping criterion and update the
                cluster centers.
            pretrain_epochs: Number of epochs for pretraining DEC.
            alpha_k: The factor to control the penalty term of the number of clusters.
            max_train_samples: Number of randomly selected rows to train the DEC.

        Examples:
            >>> df = pd.read_csv("path/to/dataframe")
            >>> result = AAIClusteringTask().run(
            ...     df,
            ...     ["feature1", "feature2", "feature3"]
            ... )
            >>> result

        Returns:
            Dict: Dictionnary containing the result
                - "status": "SUCCESS" if the task successfully ran else "FAILURE"
                - "messenger": Message returned with the task
                - "data": Dictionary containing the data for the clustering task
                    - "cluster_id": ID of the generated cluster
                    - "explanation": Explanation for the points for this cluster
                    - "encoded_value": Encoded value for centroid for this cluster
                    - "projected_value": Projected centroid for this cluster
                    - "projected_nearest_point": Nearest point for the centroid
                - "data_v2": Updated dictionary containing the data for the clustering task
                    - "clusters": Same dictionary as data
                    - "shap_values": Shapley values for clustering
                - "runtime": Time taken to run the task
                - "validations": List of validations on the data,
                    non-empty if the data presents a problem for the task
        """
        import tensorflow as tf

        # This needs to be done to make the shap library compatible
        tf.compat.v1.disable_v2_behavior()

        import time
        import logging
        import pandas as pd
        import numpy as np
        from collections import defaultdict
        from sklearn.impute import SimpleImputer
        from sklearn.tree import DecisionTreeClassifier
        from actableai.data_validation.params import ClusteringDataValidator
        from actableai.data_validation.base import CheckLevels
        from actableai.utils.preprocessors.preprocessing import impute_df
        from actableai.utils import handle_boolean_features
        from actableai.clustering.transform import ClusteringDataTransformer
        from actableai.clustering.explain import generate_cluster_descriptions
        from actableai.clustering.models import model_dict as clustering_model_dict
        from actableai.clustering.models.base import Model as ClusteringModel
        from actableai.embedding.models import model_dict as projection_model_dict
        from actableai.embedding.models.base import Model as ProjectionModel

        from actableai.utils.sanitize import sanitize_timezone

        pd.set_option("chained_assignment", "warn")
        start = time.time()

        # To resolve any issues of access rights make a copy
        df = df.copy()
        df = sanitize_timezone(df)
        df = handle_boolean_features(df)

        if features is None:
            features = list(df.columns)
        if max_train_samples is None:
            max_train_samples = len(df)

        df_train = df[features]

        parameters_validation = None
        parameters_definition = self.get_parameters()
        if parameters is None or len(parameters) <= 0:
            parameters = parameters_definition.get_default()
        else:
            (
                parameters_validation,
                parameters,
            ) = parameters_definition.validate_process_parameter(parameters)

        clustering_model_class = None
        clustering_model_parameters = None
        projection_model_class = None
        projection_model_parameters = None
        if parameters_validation is None or len(parameters_validation) == 0:
            clustering_model_name, clustering_model_parameters = next(
                iter(parameters["clustering_model"].items())
            )
            clustering_model_class = clustering_model_dict[
                ClusteringModel[clustering_model_name]
            ]

            projection_model_name, projection_model_parameters = next(
                iter(parameters["projection_model"].items())
            )
            projection_model_class = projection_model_dict[
                ProjectionModel[projection_model_name]
            ]

        data_validation_results = ClusteringDataValidator().validate(
            features,
            df_train,
            n_cluster=num_clusters,
            explain_samples=explain_samples,
            max_train_samples=max_train_samples,
            clustering_model_class=clustering_model_class,
        )

        if parameters_validation is not None:
            data_validation_results += parameters_validation.to_check_results(
                name="ParametersChecker",
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

        df_train = df_train.dropna(how="all", axis=1)
        features = list(df_train.columns)

        numeric_columns = list(df_train.select_dtypes(include=["number"]).columns)
        if len(numeric_columns):
            df_train[numeric_columns] = pd.DataFrame(
                SimpleImputer(strategy="median").fit_transform(
                    df_train[numeric_columns]
                ),
                columns=numeric_columns,
            )
        categorical_columns = list(df_train.select_dtypes(exclude=["number"]).columns)
        if len(categorical_columns):
            df_train[categorical_columns] = df_train[categorical_columns].fillna("None")

        # Process data
        preprocessor = ClusteringDataTransformer(drop_low_info=drop_low_info)
        transformed_values = preprocessor.fit_transform(df_train)
        if transformed_values is None:
            return {
                "status": "FAILURE",
                "validations": [
                    {
                        "name": "Data Transformation",
                        "level": "CRITICAL",
                        "message": "Every columns have only unique values and cannot be used for clustering",
                    }
                ],
                "runtime": time.time() - start,
                "data": {},
            }

        sampled_transformed_values = transformed_values
        if max_train_samples is not None:
            max_train_samples = min(max_train_samples, transformed_values.shape[0])
            sampled_transformed_values = (
                pd.DataFrame(transformed_values).sample(max_train_samples).values
            )

        clustering_model = clustering_model_class(
            input_size=transformed_values.shape[-1],
            num_clusters=num_clusters,
            parameters=clustering_model_parameters,
            process_parameters=False,
            verbosity=0,
        )

        clustering_model.fit(sampled_transformed_values)
        cluster_ids = clustering_model.predict(transformed_values)
        projection = clustering_model.project(transformed_values)

        shap_values = []
        if explain_samples and not clustering_model.has_explanations:
            logging.warning("Model does not support explanations")
        elif explain_samples:
            shap_values = clustering_model.explain_samples(transformed_values)

            row_index, column_index = np.meshgrid(
                np.arange(shap_values.shape[1]), np.arange(shap_values.shape[2])
            )
            shap_values = shap_values[cluster_ids, row_index, column_index].transpose(
                1, 0
            )

            df_final_shap_values = pd.DataFrame(
                0, index=np.arange(transformed_values.shape[0]), columns=features
            )

            for i, feature in enumerate(features):
                df_final_shap_values[feature] = shap_values[
                    :, preprocessor.feature_links[i]
                ].sum(axis=1)

            shap_values = df_final_shap_values

        projection_model = projection_model_class(
            embedding_size=2,
            parameters=projection_model_parameters,
            process_parameters=False,
            verbosity=0,
        )

        x_embedded = projection_model.fit_transform(projection, cluster_ids)

        # Return data
        data = []
        points_x = x_embedded[:, 0]
        points_y = x_embedded[:, 1]

        origin_dict = df_train.to_dict("record")
        for idx, (i, j, k, l) in enumerate(
            zip(points_x.tolist(), points_y.tolist(), cluster_ids, origin_dict)
        ):
            data.append((k, {"x": i, "y": j}, l))

        res = defaultdict(list)
        for idx, (k, v, s) in enumerate(data):
            res[k].append({"train": v, "column": s, "index": df_train.index[idx]})

        # Explain clusters
        clusters = [{"cluster_id": int(k), "value": v} for k, v in res.items()]
        rows, cluster_id = [], []
        for c in clusters:
            for row in c["value"]:
                rows.append(row["column"])
                cluster_id.append(c["cluster_id"])
        df_ = pd.DataFrame(rows)
        impute_df(df_)
        df_dummies = pd.get_dummies(df_)
        dummy_columns = set(df_dummies.columns) - set(df_.columns)
        clf = DecisionTreeClassifier(
            max_depth=cluster_explain_max_depth,
            min_impurity_decrease=cluster_explain_min_impurity_decrease,
            min_samples_leaf=cluster_explain_min_samples_leaf / len(clusters),
        )
        clf.fit(df_dummies, cluster_id)
        cluster_explanations = generate_cluster_descriptions(
            clf.tree_,
            df_dummies.columns,
            dummy_columns,
            min_precision=cluster_explain_min_precision,
        )

        for cluster in clusters:
            cid = cluster["cluster_id"]
            cluster["explanation"] = "\n".join(cluster_explanations[cid])

        runtime = time.time() - start

        return {
            "data_v2": {
                "clusters": clusters,
                "shap_values": shap_values,
            },
            "data": clusters,
            "status": "SUCCESS",
            "messenger": "",
            "runtime": runtime,
            "validations": [
                {"name": x.name, "level": x.level, "message": x.message}
                for x in failed_checks
            ],
        }
