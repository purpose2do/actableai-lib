class AAIModelInference:
    """
    TODO write documentation
    """

    @classmethod
    def deploy(cls, num_replicas, ray_options, s3_bucket, s3_prefix=""):
        """
        TODO write documentation
        """
        from ray import serve

        return serve.deployment(
            cls,
            name=cls.__name__,
            num_replicas=num_replicas,
            ray_actor_options=ray_options,
            init_args=(s3_bucket, s3_prefix),
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

    def __init__(self, s3_bucket, s3_prefix=""):
        """
        TODO write documentation
        """
        import boto3

        self.task_models = {}
        self.s3_prefix = s3_prefix

        s3 = boto3.resource("s3")
        self.bucket = s3.Bucket(s3_bucket)

    def _load_model(self, task_id):
        """
        TODO write documentation
        """
        import os
        import pickle
        from botocore.exceptions import ClientError

        path = os.path.join(self.s3_prefix, task_id, "model.p")
        obj = self.bucket.Object(path)
        try:
            raw_model = obj.get()["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return False
            else:
                raise

        self.task_models[task_id] = pickle.loads(raw_model)
        return True

    def _get_model(self, task_id, raise_error=True):
        """
        TODO write documentation
        """
        from actableai.exceptions.models import InvalidTaskIdError

        if task_id not in self.task_models and not self._load_model(task_id):
            if raise_error:
                raise InvalidTaskIdError()
            else:
                return None
        return self.task_models[task_id]

    def predict(
        self,
        task_id,
        df,
        return_probabilities=False,
        probability_threshold=0.5,
        positive_label=None,
    ):
        """
        TODO write documentation
        """
        from actableai.exceptions.models import InvalidPositiveLabelError

        df_proba = self.predict_proba(task_id, df)

        class_labels = list(df_proba.columns)

        # Regression
        if len(class_labels) <= 1:
            return df_proba

        # Multiclass
        if len(class_labels) > 2:
            df = df_proba.idxmax(axis="columns")
            if return_probabilities:
                return df, df_proba
            return df

        # Binary
        if positive_label is None:
            positive_label = class_labels[1]

        if positive_label not in class_labels:
            raise InvalidPositiveLabelError()

        negative_label = list(set(class_labels).difference({positive_label}))[0]

        df_true_label = df_proba[positive_label] >= probability_threshold
        df_true_label = df_true_label.astype(str).map(
            {"True": positive_label, "False": negative_label}
        )

        if return_probabilities:
            return df_true_label, df_proba
        return df_true_label

    def predict_proba(self, task_id, df):
        """
        TODO write documentation
        """
        import pandas as pd
        from actableai.exceptions.models import MissingFeaturesError

        task_model = self._get_model(task_id)

        # FIXME this considers that the model we have is an AutoGluon which will not
        #  be the case in the future
        feature_generator = task_model._learner.feature_generator
        first_feature_links = feature_generator.get_feature_links()

        missing_features = list(
            set(first_feature_links.keys()).difference(set(df.columns))
        )

        if len(missing_features) > 0:
            raise MissingFeaturesError(missing_features=missing_features)

        df_proba = task_model.predict_proba(df, as_multiclass=True)
        if isinstance(df_proba, pd.Series):
            df_proba = df_proba.to_frame()
        return df_proba

    def get_metadata(self, task_id):
        """
        TODO write documentation
        """
        task_model = self._get_model(task_id)

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
        }

        if problem_type == "multiclass" or problem_type == "binary":
            class_labels = task_model.class_labels
            metadata["class_labels"] = class_labels

        return metadata
