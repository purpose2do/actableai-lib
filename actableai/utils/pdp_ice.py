from sklearn.inspection import partial_dependence
import logging

from actableai.causal.predictors import SKLearnTabularWrapper
from actableai.utils import get_type_special


def _compute_pdp_ice(model_sk, df_train, feature, kind, grid_resolution):
    """
    Compute Partial Dependence Plot (PDP) and Individual Conditional Expectation
    (ICE) for a given sklearn model and feature (column).
    Note: Categorical features only partially supported; when using
        return_type='plot',
    categorical features are converted to numerical - however, the model should
    also have been trained using the numerical features.
    Note: Categorical features used for two-way PDP may not fully function.
    It is recommended to use scikit-learn >= 1.2 for better support of
        categorical features.

    Parameters:
    model_sk (scikit-learn model): trained sklearn model on which to compute
        PDP/ICE
    df_train (pandas DataFrame): dataset on which to compute the PDP/ICE
    feature (str): name of the feature (column) on which to compute PDP/ICE
    kind (str): 'average' (PDP), 'individual' (ICE), or 'both' (pdp and ICE)
    grid_resolution (int): number of points to sample in the grid and plot
        (x-axis values)

    Returns:
    If return_type='raw':
    Dictionary-like object, with the attributes 'values' (The values with
        which the grid has been created), 'average' (PDP results) and
        'individual' (ICE results)
    If return_type='plot':
    sklearn.inspection.PartialDependenceDisplay object containing the plot.
        Raw values can be accessed from the 'pd_results' attribute.
    """

    # Check feature type; if 'mixed', convert to string
    feat_type = get_type_special(df_train[feature])
    if feat_type == "mixed":
        df_train[feature] = df_train[feature].astype(str)

    # If categorical, set grid resolution to the number of unique elements + 1
    # Note: This might take a while if there is a large number of categories
    # Otherwise (if numerical), keep the original resolution
    if feat_type not in ["integer", "numeric"]:
        grid_resolution_feature = len(df_train[feature].unique()) + 1
    else:
        grid_resolution_feature = grid_resolution

    # Drop any rows where feature value is NaN/null
    df_train = df_train.dropna(subset=[feature])

    res = partial_dependence(
        model_sk,
        df_train,
        features=feature,
        kind=kind,
        grid_resolution=grid_resolution_feature,
    )

    res["feature_type"] = feat_type

    return res


# MAIN function
def get_pdp_and_ice(
    model,
    df_train,
    features="all",
    pdp=True,
    ice=True,
    grid_resolution=100,
    verbosity=0,
    n_samples=None,
):
    """
    Get Partial Dependence Plot (PDP) and/or Individual Conditional Expectation
    (ICE) for a given model and dataframe.

    Parameters:
    model: The trained model from AAIRegressionTask() or AAIClassificationTask()
    df_train (pandas DataFrame): dataset on which to compute the PDP/ICE
    features (list or str, optional): list of feature names/column numbers on
        which to compute PDP/ICE, or 'all' to use all columns. If only one
        fetaure is required, its name or column number should be in a list.
    pdp (bool, optional): set to True to compute PDP
    ICE (bool, optional): set to True to compute ICE
    grid_resolution (int, optional): number of points to sample in the grid
        and plot (x-axis values)
    verbosity (int, optional): 0 for no output, 1 for summary output, 2 for
        detailed output
    n_samples (int, optional): The number of rows to sample in df_train. If 'None,
        no sampling is performed.

    Returns:
    A dictionary with keys as feature names and values as the computed PDP/ICE
        results
    If return_type='raw':
    tuple of two numpy arrays. First array represents the feature values and
        second array represents the model predictions

    If return_type='plot':
    sklearn.inspection.PartialDependenceDisplay object containing the plot
    """

    model_type = model.predictor.problem_type
    if model_type not in ["regression", "quantile", "multiclass", "binary"]:
        raise ValueError(
            "'model_type' must be either 'regression' (when doing regression), \
            'quantile' (when doing quantile regression)), 'classifier' (when \
            doing classification), or 'binary' (when doing binary classification)"
        )

    model_sk = SKLearnTabularWrapper(model.predictor)
    if verbosity > 1:
        logging.info("Model wrapped with sklearn wrapper")

    if features == "all":  # Use all columns on which the model was trained
        features = model.predictor.features()

    if n_samples is not None:
        # Only sample if there are more rows than the requested number of samples
        if len(df_train) > n_samples:
            df_train = df_train.sample(
                n=n_samples, axis=0, replace=True, random_state=1
            )
            if verbosity > 1:
                logging.info(f"Sampled {n_samples} rows from the dataset")

    for feature in df_train.columns:
        # Check if any column contains only empty values
        if df_train[feature].isnull().all():
            if verbosity > 0:
                logging.info(
                    f'All rows in the column "{feature}" are null; replacing with 0'
                )
            df_train[feature] = df_train[feature].fillna(0, inplace=False)

    # Iterate over each column/feature
    res_all = dict()
    for feature in features:
        if verbosity > 0:
            logging.info("")
            if type(feature) == int:
                feature_name = df_train.columns[feature]
                logging.info(f"Feature: {feature} ({feature_name})")
            else:
                logging.info(f"Feature: {feature}")

        # Determine 'kind' (plot both PDP and ICE, or only one of them)
        if pdp and ice:
            kind = "both"
            kind_str = "PDP and ICE"
        elif pdp:
            kind = "average"
            kind_str = "PDP"
        elif ice:
            kind = "individual"
            kind_str = "ICE"
        else:
            raise ValueError(
                "Both 'pdp' and 'ice' are set to 'False'; at least one of 'pdp' or \
                'ice' must be 'True'!"
            )

        if verbosity > 1:
            logging.info("Parameter 'kind' set to:", kind)
            logging.info(f"Performing {kind_str}")

        res_all[feature] = _compute_pdp_ice(
            model_sk,
            df_train,
            feature,
            kind,
            grid_resolution,
        )

    return res_all
