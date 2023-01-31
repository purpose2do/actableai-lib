
from actableai.utils.categorical_numerical_convert import convert_categorical_to_num

def pdp_sklearn_wrapper(model, model_type):
  """
  This function takes in a model object created by AutoGluon and wraps it with a sklearn wrapper, allowing it to be used for computation of the PDP and/or ICE. The model type can be set as either 'regressor' or 'classifier'.

  Input:
  model (AutoGluon object): The model to be wrapped.
  model_type (str): The type of the model, either 'regressor' or 'classifier'.

  Returns:
  model_sk3 (SKLearnTabularWrapper): The wrapped model, ready to be used with plot_partial_dependence.

  """

  if ((model_type!='regressor') and (model_type!='classifier')):
    raise ValueError("'model_type' must be either 'regressor' (when doing regression) or 'classifier' (when doing classification)")

  from actableai.causal.predictors import SKLearnTabularWrapper

  # Wrap AutoGluon object with sklearn wrapper
  model_sk3 = SKLearnTabularWrapper(model.predictor)

  # Bypass error by sklearn that model is not yet fitted
  model_sk3.ag_predictor_ = model_sk3.ag_predictor

  # Type of model ('regressor' or 'classifier')
  # is_regressor check: https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b611bf873bd5836748647221480071a87/sklearn/base.py#L1017
  # is_classifier check: https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b611bf873bd5836748647221480071a87/sklearn/base.py#L1001
  model_sk3._estimator_type = model_type ###

  # Bypass error that estimator.classes_[0] does not exist
  model_sk3.classes_ = [0]

  return model_sk3


def compute_pdp_ice(model_sk, 
                    df_train,
                    return_type,
                    feature,
                    kind,
                    grid_resolution,
                    drop_invalid=True):
  
  """
  Compute Partial Dependence Plot (PDP) and Individual Conditional Expectation 
  (ICE) for a given sklearn model and feature (column).
  Note: Categorical features only partially supported; when using return_type='plot', 
  categorical features are converted to numerical - however, the model should 
  also have been trained using the numerical features. 
  Note: Categorical features used for two-way PDP may not fully function.
  It is recommended to use scikit-learn >= 1.2 for better support of categorical features.

  Parameters:
  model_sk (scikit-learn model): trained sklearn model on which to compute PDP/ICE
  df_train (pandas DataFrame): dataset on which to compute the PDP/ICE
  return_type (str): 'raw' to return the raw PDP/ICE data, 'plot' to return a plot of the data
  feature (str): name of the feature (column) on which to compute PDP/ICE
  kind (str): 'average' (PDP), 'individual' (ICE), or 'both' (PDP and ICE)
  grid_resolution (int): number of points to sample in the grid and plot (x-axis values)
  drop_invalid (bool, optional): Whether to drop rows containing NaNs or Inf values in any of the columns

  Returns:
  If return_type='raw':
  Dictionary-like object, with the attributes 'values' (The values with which the grid has been created), 'average' (PDP results) and 'individual' (ICE results)
  If return_type='plot':
  sklearn.inspection.PartialDependenceDisplay object containing the plot. Raw values can be accessed from the 'pd_results' attribute.
  """

  import numpy as np
    
  # Drop any rows containing NaNs or Infs
  if drop_invalid:
    #s1 = df_train.shape
    df_train = df_train.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    #s2 = df_train.shape
    # If s2[0]<s1[0], then some rows were dropped

  if return_type=='raw':
    from sklearn.inspection import partial_dependence
    return partial_dependence(model_sk, df_train, 
                              features=feature, 
                              kind=kind,
                              grid_resolution=grid_resolution)
                              #categorical_features=['location', 'neighborhood']) # Available in scikit-learn >= 1.2

  elif return_type=='plot':
    from sklearn.inspection import PartialDependenceDisplay
    return PartialDependenceDisplay.from_estimator(model_sk, df_train, 
                                            features=[feature], 
                                            kind=kind,
                                            grid_resolution=grid_resolution)
                                            #categorical_features=['location', 'neighborhood']) # Available in scikit-learn >= 1.2
                                            #centered=True) # Available in scikit-learn >= 1.1 (https://scikit-learn.org/stable/modules/generated/sklearn.inspection.PartialDependenceDisplay.html#sklearn.inspection.PartialDependenceDisplay.from_estimator:~:text=centering%20is%20done.-,New%20in%20version%201.1.,-subsamplefloat%2C%20int)

# MAIN function
def get_pdp_and_ice(model, 
                    model_type, 
                    df_train, 
                    features='all', 
                    PDP=True, 
                    ICE=True, 
                    return_type='raw',
                    grid_resolution=100,
                    verbosity=0,
                    plot_convert_to_num=True,
                    drop_invalid=True):
  
  """
  Get Partial Dependence Plot (PDP) and/or Individual Conditional Expectation (ICE) for a given model and dataframe.

  Parameters:
  model: The trained model from AAIRegressionTask() or AAIClassificationTask()
  model_type (str): 'regressor' (when doing regression (using 'AAIRegressionTask()')) or 'classifier' (when doing classification (using 'AAIClassificationTask()')).
  df_train (pandas DataFrame): dataset on which to compute the PDP/ICE
  features (list or str, optional): list of feature names/column numbers on which to compute PDP/ICE, or 'all' to use all columns. If only one fetaure is required, its name or column number should be in a list.
  PDP (bool, optional): set to True to compute PDP
  ICE (bool, optional): set to True to compute ICE
  return_type (str, optional): 'raw' to return the raw PDP/ICE data, 'plot' to return a plot of the data
  grid_resolution (int, optional): number of points to sample in the grid and plot (x-axis values)
  verbosity (int, optional): 0 for no output, 1 for summary output, 2 for detailed output
  plot_convert_to_num (bool, optional): Flag to determine if any categorical features in the dataframe should be enumerated. This should be done if using kind='plot' and the dataframe has not already been converted. However, it should be noted that the trained model should have used the converted dataframe, in order to ensure that the PDP/ICE results are correct.
  drop_invalid (bool, optional): Whether to drop rows containing NaNs or Inf values in any of the columns

  Returns:
  A dictionary with keys as feature names and values as the computed PDP/ICE results
  If return_type='raw':
  tuple of two numpy arrays. First array represents the feature values and second array represents the model predictions
  
  If return_type='plot':
  sklearn.inspection.PartialDependenceDisplay object containing the plot
  """

  # Determine 'kind' (plot both PDP and ICE, or only one of them)
  if PDP and ICE:
    kind='both'  
    kind_str = 'PDP and ICE'
  elif PDP:
    kind='average'  
    kind_str = 'PDP'
  elif ICE:
    kind='individual'
    kind_str = 'ICE'
  else:
    raise ValueError("Both 'PDP' and 'ICE' are set to 'False'; at least one of 'PDP' or 'ICE' must be 'True'!")

  if ((model_type!='regressor') and (model_type!='classifier')):
    raise ValueError("'model_type' must be either 'regressor' (when doing regression) or 'classifier' (when doing classification)")

  if verbosity>1:
    print("Parameter 'kind' set to:", kind)
  
  if verbosity>1:
    print('Applying wrapper')
  model_sk = pdp_sklearn_wrapper(model, model_type)
  if verbosity>1:
    print('Model wrapped with sklearn wrapper')

  if features=='all': # Use all columns
    features = df_train.columns

  res_all = dict()
  for feature in features:  

    if verbosity>0: 
      if type(feature)==int:
        feature_name = df_train.columns[feature]
        print(f'Performing {kind_str} for feature {feature} ({feature_name})')
      else:
        print(f'Performing {kind_str} for {feature}')

    # Convert categorical features to numerical if using 'plot'
    # NOTE: Model should have been trained with the converted features to ensure correct computation of the PDP/ICE!
    if ((return_type=='plot') and (plot_convert_to_num)):
      df_train, uniques_all = convert_categorical_to_num(df_train)
    
    res_all[feature] = compute_pdp_ice(model_sk,
                                       df_train,
                                       return_type,
                                       feature,
                                       kind,
                                       grid_resolution,
                                       drop_invalid=drop_invalid)
  
  return res_all


