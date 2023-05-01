from functools import lru_cache
from typing import Dict, List, Any

import ray
from ray import serve
import pandas as pd

from actableai.exceptions.models import UnknownModelClassError
from actableai.models.intervention import empty_string_to_nan


class AAIModelInferenceHead:
    @classmethod
    def get_actor(cls, cache_maxsize: int = 20):
        head_actor = None

        try:
            head_actor = ray.get_actor(name=cls.__name__)
        except ValueError:
            head_actor = (
                ray.remote(cls)
                .options(name=cls.__name__, lifetime="detached")
                .remote(cache_maxsize=cache_maxsize)
            )

        return head_actor

    def __init__(self, cache_maxsize: int = 20):
        self._load_model_ref_cached = lru_cache(maxsize=cache_maxsize)(
            self._load_model_ref
        )

    @staticmethod
    def load_raw_model(s3_bucket_name, path):
        import boto3
        from botocore.exceptions import ClientError

        # Load from S3
        s3 = boto3.resource("s3")
        bucket = s3.Bucket(s3_bucket_name)

        obj = bucket.Object(path)
        try:
            raw_model = obj.get()["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            else:
                raise

        return raw_model

    @classmethod
    def _load_model_ref(cls, s3_bucket_name, path):
        import dill as pickle

        raw_model = cls.load_raw_model(s3_bucket_name, path)
        if raw_model is None:
            return None

        model = pickle.loads(raw_model)
        model_ref = ray.put(model)

        return model_ref

    async def get_model_ref(self, s3_bucket_name, path):
        return self._load_model_ref_cached(s3_bucket_name=s3_bucket_name, path=path)


class AAIModelInference:
    """
    TODO write documentation
    """

    @classmethod
    def deploy(
        cls,
        ray_autoscaling_configs,
        ray_options,
        s3_bucket,
        s3_prefix="",
        cache_maxsize=5,
        head_cache_maxsize=20,
    ):
        """
        TODO write documentation
        """
        from ray import serve

        return serve.deployment(
            cls,
            name=cls.__name__,
            autoscaling_config=ray_autoscaling_configs,
            ray_actor_options=ray_options,
            init_args=(s3_bucket, s3_prefix, cache_maxsize, head_cache_maxsize),
        ).deploy()

    @classmethod
    def get_handle(cls):
        """
        TODO write documentation
        """
        return cls.get_deployment().get_handle()

    @classmethod
    def get_deployment(cls):
        """
        TODO write documentation
        """
        from ray import serve

        return serve.get_deployment(cls.__name__)

    def __init__(
        self,
        s3_bucket,
        s3_prefix="",
        cache_maxsize=5,
        head_cache_maxsize=20,
    ):
        """
        TODO write documentation
        """
        self.s3_prefix = s3_prefix
        self.s3_bucket_name = s3_bucket

        self.head_cache_maxsize = head_cache_maxsize
        self._load_model_cached = lru_cache(maxsize=cache_maxsize)(self._load_model)

        # This is done to start the actor if it did not start already
        AAIModelInferenceHead.get_actor(cache_maxsize=self.head_cache_maxsize)

    @staticmethod
    def get_model_path(s3_prefix, task_id):
        """
        TODO write documentation
        """
        import os

        return os.path.join(s3_prefix, task_id, "model.p")

    @staticmethod
    def _load_model(head_cache_maxsize, s3_bucket_name, path):
        head_actor = AAIModelInferenceHead.get_actor(cache_maxsize=head_cache_maxsize)

        model_ref = ray.get(
            head_actor.get_model_ref.remote(
                s3_bucket_name=s3_bucket_name,
                path=path,
            )
        )

        if model_ref is None:
            return None
        return ray.get(model_ref)

    def _get_model(self, task_id, raise_error=True):
        """
        TODO write documentation
        """
        from actableai.exceptions.models import InvalidTaskIdError

        model = self._load_model_cached(
            head_cache_maxsize=self.head_cache_maxsize,
            s3_bucket_name=self.s3_bucket_name,
            path=self.get_model_path(self.s3_prefix, task_id),
        )

        if model is None:
            if raise_error:
                raise InvalidTaskIdError()
            else:
                return None

        return model

    async def predict(
        self,
        task_id,
        df,
        return_probabilities=False,
        probability_threshold=0.5,
        positive_label=None,
    ):
        return await self._predict_batched(
            {
                "task_id": task_id,
                "df": df,
                "return_probabilities": return_probabilities,
                "probability_threshold": probability_threshold,
                "positive_label": positive_label,
            }
        )

    @serve.batch(max_batch_size=64)
    async def _predict_batched(self, data: List[Dict[str, Any]]):
        task_data = {}
        results_data_list = []

        # Group data by (task_id, ...)
        for e in data:
            task_id = e["task_id"]
            df = e["df"]
            return_probabilities = e["return_probabilities"]
            probability_threshold = e["probability_threshold"]
            positive_label = e["positive_label"]

            key = (task_id, return_probabilities, probability_threshold, positive_label)

            if key in task_data:
                start_index = len(task_data[key])

                task_data[key] = pd.concat(
                    [task_data[key], df],
                    ignore_index=True,
                )
            else:
                start_index = 0
                task_data[key] = df

            results_data_list.append(
                {
                    "key": key,
                    "start_index": start_index,
                    "end_index": start_index + len(df),
                }
            )

        results_data = {}
        for key, df in task_data.items():
            task_id, return_probabilities, probability_threshold, positive_label = key
            results_data[key] = self._predict_one(
                task_id=task_id,
                df=df,
                return_probabilities=return_probabilities,
                probability_threshold=probability_threshold,
                positive_label=positive_label,
            )

        results = []

        for result_data in results_data_list:
            key = result_data["key"]
            start_index = result_data["start_index"]
            end_index = result_data["end_index"]

            result = {}
            for data_key, df in results_data[key].items():
                result[data_key] = df.iloc[start_index:end_index]
            results.append(result)

        return results

    def _predict_one(
        self,
        task_id,
        df,
        return_probabilities=False,
        probability_threshold=0.5,
        positive_label=None,
        explain_samples=False,
    ):
        from autogluon.tabular import TabularPredictor

        from actableai.models.aai_predictor import (
            AAITabularModel,
            AAITabularModelInterventional,
        )

        task_model = self._get_model(task_id)

        # We used to pickle TabularPredictor. This check is for legacy
        if isinstance(task_model, AAITabularModel):
            pred = self._predict(
                task_model.predictor,
                df,
                return_probabilities=isinstance(
                    task_model, AAITabularModelInterventional
                )
                or return_probabilities,
                probability_threshold=probability_threshold,
                positive_label=positive_label,
            )

            if explain_samples:
                pred["predictions_shaps"] = task_model.explainer.shap_values(df)

            # Here for legacy, previously the causal model was directly in the AAITabularModel
            # Now intervention has its own custom model
            if task_model.model_version <= 1:
                if isinstance(task_model, AAITabularModelInterventional) and (
                    f"intervened_{task_model.intervened_column}" in df
                    or f"expected_{task_model.predictor.label}" in df
                ):
                    pred["intervention"] = task_model.intervention_effect(df, pred)
                return pred

            if isinstance(task_model, AAITabularModelInterventional) and (
                f"intervened_{task_model.intervention_model.current_intervention_column}"
                in df
                or f"expected_{task_model.predictor.label}" in df
            ):
                new_intervention_col = (
                    "intervened_"
                    + task_model.intervention_model.current_intervention_column
                )
                intervention_col = (
                    task_model.intervention_model.current_intervention_column
                )
                df[task_model.predictor.label] = pred["prediction"]
                df = empty_string_to_nan(
                    df,
                    task_model,
                    intervention_col,
                    new_intervention_col,
                    f"expected_{task_model.predictor.label}",
                )
                if (
                    not task_model.intervention_model.causal_model.discrete_treatment
                    and task_model.predictor.problem_type == "regression"
                ):
                    new_outcome = task_model.intervention_model.predict_two_way(df)
                    pred["intervention"] = new_outcome
                else:
                    new_outcome = task_model.intervention_model.predict(df)
                    pred["intervention"] = pd.DataFrame(
                        {
                            f"expected_{task_model.predictor.label}": new_outcome[
                                task_model.predictor.label + "_intervened"
                            ],
                            f"intervened_{task_model.intervention_model.current_intervention_column}": [
                                None for _ in range(len(df))
                            ],
                        }
                    )

            return pred
        elif isinstance(task_model, TabularPredictor):
            # Run legacy task_model directly
            return self._predict(
                task_model,
                df,
                return_probabilities,
                probability_threshold,
                positive_label,
            )
        else:
            raise UnknownModelClassError()

    def _predict(
        self,
        task_model,
        df,
        return_probabilities=False,
        probability_threshold=0.5,
        positive_label=None,
    ) -> Dict:
        """
        TODO write documentation
        """
        from actableai.exceptions.models import MissingFeaturesError
        from actableai.exceptions.models import InvalidPositiveLabelError

        result = {}

        # FIXME this considers that the model we have is an AutoGluon which will not
        #  be the case in the future
        feature_generator = task_model._learner.feature_generator
        first_feature_links = feature_generator.get_feature_links()

        missing_features = list(
            set(first_feature_links.keys()).difference(set(df.columns))
        )

        if len(missing_features) > 0:
            raise MissingFeaturesError(missing_features=missing_features)

        # Regression
        if task_model.problem_type == "regression":
            preds = task_model.predict(df)
            result["prediction"] = preds.to_frame(name=task_model.label)
            return result

        if task_model.problem_type == "quantile":
            preds = task_model.predict(df)
            result["prediction"] = pd.DataFrame(preds)
            return result

        df_proba = self._predict_proba(task_model, df)
        class_labels = list(df_proba.columns)

        # Multiclass
        if len(class_labels) > 2:
            prediction = df_proba.idxmax(axis="columns").to_frame(name=task_model.label)
            if return_probabilities:
                result["df_proba"] = df_proba
            result["prediction"] = prediction
            return result

        # Binary
        if positive_label is None:
            positive_label = class_labels[1]

        if positive_label not in class_labels:
            raise InvalidPositiveLabelError()

        negative_label = list(set(class_labels).difference({positive_label}))[0]

        df_true_label = df_proba[positive_label] >= probability_threshold
        df_true_label = (
            df_true_label.astype(str)
            .map({"True": positive_label, "False": negative_label})
            .to_frame(name=task_model.label)
        )

        if return_probabilities:
            result["df_proba"] = df_proba
        result["prediction"] = df_true_label
        return result

    async def predict_proba(self, task_id, df):
        return await self._predict_proba_batched({"task_id": task_id, "df": df})

    @serve.batch(max_batch_size=64)
    async def _predict_proba_batched(self, data: List[Dict[str, Any]]):
        task_data = {}
        results_data_list = []

        # Group data by task id
        for e in data:
            task_id = e["task_id"]
            df = e["df"]

            if task_id in task_data:
                start_index = len(task_data[task_id])

                task_data[task_id] = pd.concat(
                    [task_data[task_id], df],
                    ignore_index=True,
                )
            else:
                start_index = 0
                task_data[task_id] = df

            results_data_list.append(
                {
                    "task_id": task_id,
                    "start_index": start_index,
                    "end_index": start_index + len(df),
                }
            )

        results_data = {
            task_id: self._predict_proba(task_id, df)
            for task_id, df in task_data.items()
        }

        results = []

        for result_data in results_data_list:
            task_id = result_data["task_id"]
            start_index = result_data["start_index"]
            end_index = result_data["end_index"]

            results.append(results_data[task_id].iloc[start_index:end_index])

        return results

    def _predict_proba_one(self, task_id, df):
        from actableai.models.aai_predictor import AAITabularModel

        task_model = self._get_model(task_id)
        if isinstance(task_model, AAITabularModel):
            task_model = task_model.predictor

        return self._predict_proba(task_model, df)

    def _predict_proba(self, task_model, df):
        """
        TODO write documentation
        """
        import pandas as pd

        df_proba = task_model.predict_proba(df, as_multiclass=True)
        if isinstance(df_proba, pd.Series):
            df_proba = df_proba.to_frame(name=task_model.label)
        return df_proba

    async def get_metadata(self, task_id):
        from actableai.models.aai_predictor import (
            AAITabularModel,
            AAITabularModelInterventional,
        )

        task_model = self._get_model(task_id)
        if isinstance(task_model, AAITabularModel):
            metadata = self._get_metadata(task_model.predictor)

            if hasattr(task_model, "feature_parameters"):
                metadata["feature_parameters"] = task_model.feature_parameters

            metadata["is_explainer_available"] = (
                hasattr(task_model, "explainer") and task_model.explainer is not None
            )

            if isinstance(task_model, AAITabularModelInterventional):
                if task_model.model_version <= 1:
                    metadata["intervened_column"] = task_model.intervened_column
                    metadata["discrete_treatment"] = task_model.discrete_treatment
                else:
                    metadata[
                        "intervened_column"
                    ] = task_model.intervention_model.current_intervention_column
                    metadata[
                        "discrete_treatment"
                    ] = task_model.intervention_model.causal_model.discrete_treatment
            return metadata
        return self._get_metadata(task_model)

    def _get_metadata(self, task_model):
        """
        TODO write documentation
        """

        # FIXME this considers that the model we have is an AutoGluon which will not
        #  be the case in the future
        feature_generator = task_model._learner.feature_generator
        first_feature_links = feature_generator.get_feature_links()
        features = list(first_feature_links.keys())

        problem_type = task_model.problem_type

        metadata = {
            "features": features,
            "problem_type": problem_type,
            "class_labels": None,
            "prediction_target": task_model.label,
            "quantile_levels": task_model.quantile_levels,
        }

        if problem_type == "multiclass" or problem_type == "binary":
            class_labels = task_model.class_labels
            metadata["class_labels"] = class_labels

        return metadata

    async def is_model_available(self, task_id):
        """
        TODO write documentation
        """
        import boto3

        s3_client = boto3.client("s3")
        object_list = s3_client.list_objects_v2(
            Bucket=self.s3_bucket_name,
            Prefix=self.get_model_path(self.s3_prefix, task_id),
        )

        return "Contents" in object_list and len(object_list["Contents"]) > 0
