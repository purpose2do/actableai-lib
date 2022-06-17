import numpy as np
from typing import Dict


def cross_validation_curve(
    cross_val_auc_curves: Dict,
    x: str = "False Positive Rate",
    y: str = "True Positive Rate",
) -> Dict:
    """Computes the combined ROC curves for cross validation

    Args:
        roc_curves_dictionnary: A dictionnary containing the ROC curves for each classifier
    Returns:
        A dictionnary containing the combined ROC curves for each classifier
    """
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    first_key = list(cross_val_auc_curves.keys())[0]
    for i, _ in enumerate(cross_val_auc_curves[first_key]):
        interp_tpr = np.interp(
            mean_fpr,
            cross_val_auc_curves[x][i],
            cross_val_auc_curves[y][i],
        )
        tprs.append(interp_tpr)
    mean_tpr = np.mean(tprs, axis=0)
    return {
        x: mean_fpr,
        y: mean_tpr,
        y + " std err": np.std(tprs, axis=0),
        "thresholds": mean_fpr,
        "positive_label": cross_val_auc_curves["positive_label"][0],
        "negative_label": cross_val_auc_curves["negative_label"][0],
    }
