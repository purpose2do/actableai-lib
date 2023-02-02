import numpy as np
from actableai.utils.categorical_numerical_convert import convert_categorical_to_num
from actableai.causal.predictors import SKLearnTabularWrapper


def _compute_pdp_ice(
    model_sk, df_train, return_type, feature, kind, grid_resolution, drop_invalid=True
):
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
    return_type (str): 'raw' to return the raw pdp/ICE data, 'plot' to return
        a plot of the data
    feature (str): name of the feature (column) on which to compute PDP/ICE
    kind (str): 'average' (PDP), 'individual' (ICE), or 'both' (pdp and ICE)
    grid_resolution (int): number of points to sample in the grid and plot
        (x-axis values)
    drop_invalid (bool, optional): Whether to drop rows containing NaNs or
        Inf values in any of the columns

    Returns:
    If return_type='raw':
    Dictionary-like object, with the attributes 'values' (The values with
        which the grid has been created), 'average' (PDP results) and
        'individual' (ICE results)
    If return_type='plot':
    sklearn.inspection.PartialDependenceDisplay object containing the plot.
        Raw values can be accessed from the 'pd_results' attribute.
    """

    # Drop any rows containing NaNs or Infs
    if drop_invalid:
        df_train = df_train.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    if return_type == "raw":
        from sklearn.inspection import partial_dependence

        return partial_dependence(
            model_sk,
            df_train,
            features=feature,
            kind=kind,
            grid_resolution=grid_resolution,
        )

    elif return_type == "plot":
        from sklearn.inspection import PartialDependenceDisplay

        return PartialDependenceDisplay.from_estimator(
            model_sk,
            df_train,
            features=[feature],
            kind=kind,
            grid_resolution=grid_resolution,
        )


# MAIN function
def get_pdp_and_ice(
    model,
    df_train,
    features="all",
    pdp=True,
    ice=True,
    return_type="raw",
    grid_resolution=100,
    verbosity=0,
    plot_convert_to_num=True,
    drop_invalid=True,
    inplace=False,
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
    return_type (str, optional): 'raw' to return the raw PDP/ICE data, 'plot'
        to return a plot of the data
    grid_resolution (int, optional): number of points to sample in the grid
        and plot (x-axis values)
    verbosity (int, optional): 0 for no output, 1 for summary output, 2 for
        detailed output
    plot_convert_to_num (bool, optional): Flag to determine if any categorical
        features in the dataframe should be enumerated. This should be done if
            using kind='plot' and the dataframe has not already been converted.
            However, it should be noted that the trained model should have used
            the converted dataframe, in order to ensure that the PDP/ICE
            results are correct.
    drop_invalid (bool, optional): Whether to drop rows containing NaNs or Inf
        values in any of the columns
    inplace (bool, optional): Whether to perform modifications to df_train in-place
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
            'quantile' (when doing quantiel regression)), 'classifier' (when \
            doing classification), or 'binary' (when doing binary classification)"
        )

    model_sk = SKLearnTabularWrapper(model.predictor)
    if verbosity > 1:
        print("Model wrapped with sklearn wrapper")

    if features == "all":  # Use all columns
        features = df_train.columns

    if n_samples is not None:
        # Only sample if there are more rows than the requested numebr of samples
        if len(df_train)>n_samples:
            df_train = df_train.sample(n=n_samples, axis=0, replace=True, random_state=1)
            if verbosity > 1:
                print(f"Sampled {n_samples} rows from the dataset")

    # Convert categorical features to numerical if using 'plot'
    # NOTE: Model should have been trained with the converted features to
    # ensure correct computation of the PDP/ICE!
    if (return_type == "plot") and (plot_convert_to_num):
        df_train, uniques_all = convert_categorical_to_num(df_train, inplace=inplace)
        if verbosity > 1:
            print(
                "Categorical features converted to numerical. Ensure that \
          the model was also trained with numerical values!"
            )

    res_all = dict()
    for feature in features:
        if verbosity > 0:
            if type(feature) == int:
                feature_name = df_train.columns[feature]
                print(f"Feature: {feature} ({feature_name})")
            else:
                print(f"Feature: {feature}")

        # Check if attempting to use 'plot' with 2-way ICE
        ice_feat = ice
        if (return_type == "plot") and (type(feature) == tuple):
            if verbosity > 0:
                "Two-way ICE cannot be performed when return_type=='plot'; setting ICE to False"
            ice_feat = False

        # Determine 'kind' (plot both PDP and ICE, or only one of them)
        if pdp and ice_feat:
            kind = "both"
            kind_str = "PDP and ICE"
        elif pdp:
            kind = "average"
            kind_str = "PDP"
        elif ice_feat:
            kind = "individual"
            kind_str = "ICE"
        else:
            raise ValueError(
                "Both 'pdp' and 'ice' are set to 'False'; at least one of 'pdp' or \
                'ice' must be 'True'!"
            )

        if verbosity > 1:
            print("Parameter 'kind' set to:", kind)
            print(f"Performing {kind_str}")

        res_all[feature] = _compute_pdp_ice(
            model_sk,
            df_train,
            return_type,
            feature,
            kind,
            grid_resolution,
            drop_invalid=drop_invalid,
        )

    return res_all
