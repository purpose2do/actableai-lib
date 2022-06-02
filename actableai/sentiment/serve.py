class AAISentimentExtractor:
    """
    TODO write documentation
    """

    @classmethod
    def deploy(
        cls,
        num_replicas,
        ray_options,
    ):
        """
        TODO write documentation
        """
        from ray import serve

        return serve.deployment(
            cls,
            name=cls.__name__,
            num_replicas=num_replicas,
            ray_actor_options=ray_options,
            init_args=(),
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

    def __init__(self) -> None:
        """
        TODO write documentation
        """
        import multi_rake
        from pyabsa import APCCheckpointManager

        self.rake = multi_rake.Rake(min_freq=1)
        self.sent_classifier = APCCheckpointManager.get_sentiment_classifier(
            checkpoint="multilingual2",
            auto_device=True,  # Use CUDA if available
        )

    def predict(self, X, rake_threshold=1.0):
        """
        TODO write documentation
        """
        keywords, candidates = self.rake.apply_sentences(X)
        keywords = set([kw[0].text for kw in keywords if kw[1] >= rake_threshold])
        results = []
        for candidate in candidates:
            if candidate.text in keywords:
                sent = X[candidate.sentence_id]
                annotated_sent = (
                    sent[: candidate.start_position]
                    + "[ASP]"
                    + sent[candidate.start_position : candidate.end_position]
                    + "[ASP]"
                    + sent[candidate.end_position :]
                )
                sentiment = self.sent_classifier.infer(
                    annotated_sent, print_result=False)
                results.append(
                    {
                        "keyword": candidate.text,
                        "sentiment": sentiment["sentiment"][0].lower(),
                        "confidence": sentiment["confidence"][0],
                    }
                )

        return results


if __name__ == "__main__":
    sentences = [
        "The bread is top notch as well.",
        "Certainly not the best sushi in New York, however, it is always fresh, and the place is very clean, sterile.",
        "I love the drinks, esp lychee martini, and the food is also VERY good.",
        "In fact, this was not a Nicoise salad and wa`s barely eatable.",
        "While there's a decent menu, it shouldn't take ten minutes to get your drinks and 45 for a dessert pizza.",
        "Our waiter was horrible; so rude and disinterested.",
        "We enjoyed ourselves thoroughly and will be going back for the desserts ....",
        "I definitely enjoyed the food as well.#",
        "WE ENDED UP IN LITTLE ITALY IN LATE AFTERNOON.",
    ]

    m = AAISentimentExtractor()
    results = m.predict(sentences)

    print(results)
