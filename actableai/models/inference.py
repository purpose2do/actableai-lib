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
        except ClientError:
            return False

        self.task_models[task_id] = pickle.loads(raw_model)

        return True

    def predict(self, task_id, df):
        """
        TODO write documentation
        """
        if task_id not in self.task_models and not self._load_model(task_id):
            return None

        task_model = self.task_models[task_id]
        return task_model.predict(df)
