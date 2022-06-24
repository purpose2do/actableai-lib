from actableai.classification.roc_curve_cross_validation import cross_validation_curve


def test_cross_validation_curve():
    roc_curve = cross_validation_curve(
        {
            "thresholds": [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            ],
            "positive_label": ["positive", "positive"],
            "negative_label": ["negative", "negative"],
            "False Positive Rate": [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            ],
            "True Positive Rate": [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            ],
        },
        x="False Positive Rate",
        y="True Positive Rate",
    )
    assert roc_curve["False Positive Rate"].shape == (100,)
    assert roc_curve["True Positive Rate"].shape == (100,)
    assert roc_curve["thresholds"].shape == (100,)
    assert roc_curve["positive_label"] == "positive"
    assert roc_curve["negative_label"] == "negative"


def test_cross_validation_curve_with_recall():
    # len(thresholds) == len(recall) + 1 == len(precision) + 1
    roc_curve = cross_validation_curve(
        {
            "thresholds": [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            ],
            "positive_label": ["positive", "positive"],
            "negative_label": ["negative", "negative"],
            "Recall": [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            ],
            "Precision": [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            ],
        },
        x="Recall",
        y="Precision",
    )
    assert roc_curve["Recall"].shape == (100,)
    assert roc_curve["Precision"].shape == (100,)
    assert roc_curve["thresholds"].shape == (100,)
    assert roc_curve["positive_label"] == "positive"
    assert roc_curve["negative_label"] == "negative"
