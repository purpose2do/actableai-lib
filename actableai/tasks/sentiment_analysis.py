import pandas as pd
from typing import Dict

from actableai.tasks import TaskType
from actableai.tasks.base import AAITask


class AAISentimentAnalysisTask(AAITask):
    """
    Sentiment Analysis Task
    """

    @AAITask.run_with_ray_remote(TaskType.SENTIMENT_ANALYSIS)
    def run(
        self, df: pd.DataFrame, target: str, batch_size: int = 32, rake_threshold=1.0
    ) -> Dict:
        """Run a sentiment analysis on Input DataFrame

        Args:
            df: Input DataFrame
            target: Target for sentiment analysis
            batch_size: Batch Size. Defaults to 32.
            rake_threshold: Threshold for Rake scores used to extract keywords .
                Defaults to 1.0.

        Examples:
            >>> df = pd.read_csv("path/to/dataframe")
            >>> AAISentimentAnalysisTask().run(df, "target")

        Returns:
            Dict: Dictionnary of results
        """
        import math
        import time
        import ray
        from ray.exceptions import RayTaskError
        from nltk.tokenize import sent_tokenize
        from actableai.sentiment.serve import AAISentimentExtractor
        from actableai.data_validation.checkers import CheckLevels
        from multi_rake.algorithm import UnsupportedLanguageError

        start = time.time()

        # To resolve any issues of acces rights make a copy
        df = df.copy()

        # Get serve handle
        absa_handle = AAISentimentExtractor.get_handle()

        row_ids, sentences = [], []
        for i, doc in enumerate(df[target].to_list()):
            if doc is not None:
                for paragraph in doc.split("</br>"):
                    for sent in sent_tokenize(paragraph):
                        row_ids.append(i)
                        sentences.append(sent)

        # Call the deployed model batch by batch
        result = []
        unsupported_language_codes = set()
        for batch_index in range(math.ceil(len(sentences) / batch_size)):
            sentence_batch = sentences[
                batch_index * batch_size : (batch_index + 1) * batch_size
            ]
            try:
                result += ray.get(
                    absa_handle.options(method_name="predict").remote(sentence_batch)
                )
            except RayTaskError as err:
                if type(err.cause) is UnsupportedLanguageError:
                    unsupported_language_codes.add(err.cause.language_code or "Unknown")

        validations = []
        if len(unsupported_language_codes) > 0:
            validations.append(
                {
                    "name": "UnsupportedLanguage",
                    "level": CheckLevels.WARNING,
                    "message": "Unsupported language codes detected: {}".format(
                        ", ".join(unsupported_language_codes)
                    ),
                }
            )

        data = []
        for i, re in enumerate(result):
            for kw, sentiment in zip(re["keyword"], re["sentiment"]):
                data.append(
                    {
                        "keyword": kw,
                        "sentiment": sentiment,
                        "sentence": sentences[i],
                        "row": row_ids[i],
                    }
                )

        return {
            "data": data,
            "status": "SUCCESS",
            "runtime": time.time() - start,
            "message": "",
            "validations": validations,
        }
