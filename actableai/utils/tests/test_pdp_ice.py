import pandas as pd
import pytest

from actableai.utils.pdp_ice import get_pdp_and_ice
from actableai.utils.testing import unittest_hyperparameters

from actableai.tasks.regression import (
    AAIRegressionTask,
)
from actableai.tasks.classification import (
    AAIClassificationTask,
)

@pytest.fixture(scope="function")
def regression_task():
    yield AAIRegressionTask(use_ray=False)

@pytest.fixture(scope="function")
def classification_task():
    yield AAIClassificationTask(use_ray=False)

def test_pdp_ice_regression(regression_task, tmp_path):
  """
  Check if PDP and ICE for regression tasks runs without errors, and that 
  the outputs are of the expected dimensions
  """

  df_train = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": ['a', 'b', 'b', 'c', 'a', 'c', 'a', 'b', 'a', 'c'] * 2,
                "t": [2, 2, 2, 2, 2, 3, 3, 3, 4, 4] * 2,
            }
        )
  n_samples = len(df_train)

  kwargs = dict()
  kwargs["prediction_quantile_low"] = None
  kwargs["prediction_quantile_high"] = None
  kwargs["drop_duplicates"] = False
  kwargs["hyperparameters"] = unittest_hyperparameters()
  kwargs["drop_unique"] = False
  kwargs["drop_useless_features"] = False

  ### Raw ###
  # Features set to None means that we will use every single feature available
  result = regression_task.run(
      df=df_train,
      target="t",
      run_pdp=False,
      run_ice=False,
      model_directory=tmp_path,
      residuals_hyperparameters=unittest_hyperparameters(),
      presets="medium_quality_faster_train",
      **kwargs
  )


  grid_resolution = 10
  feats = ['y', 'z']
  pd_r2 = get_pdp_and_ice(result['model'],
                          'regressor', 
                          df_train, 
                          features=feats, 
                          PDP=True, 
                          ICE=True, 
                          return_type='raw', 
                          grid_resolution=grid_resolution,
                          verbosity=0,
                          drop_invalid=False)
  

  for feat_name in feats:
    n_unique = len(df_train[feat_name].unique())
    n_grid = min(n_unique, grid_resolution)
    assert(pd_r2[feat_name]['individual'].shape==(1, n_samples, n_grid))
    assert(pd_r2[feat_name]['average'].shape==(1, n_grid))
    assert(pd_r2[feat_name]['values'][0].shape==(n_grid,))


  # Check 2-way PDP
  feats = [('y', 'x')]
  pd_r2 = get_pdp_and_ice(result['model'],
                          'regressor', 
                          df_train, 
                          features=feats, 
                          PDP=True, 
                          ICE=True, 
                          return_type='raw', 
                          grid_resolution=grid_resolution,
                          verbosity=0,
                          drop_invalid=False)
  
  for feat_name in feats:
    n_unique_0 = len(df_train[feat_name[0]].unique())
    n_grid_0 = min(n_unique_0, grid_resolution)
    n_unique_1 = len(df_train[feat_name[1]].unique())
    n_grid_1 = min(n_unique_1, grid_resolution)
    assert(pd_r2[feat_name]['individual'].shape==(1, n_samples,n_grid_0, n_grid_1))
    assert(pd_r2[feat_name]['average'].shape==(1, n_grid_0, n_grid_1))
    assert(len(pd_r2[feat_name]['values'])==2)
    assert(len(pd_r2[feat_name]['values'][0]==(n_grid_0)))
    assert(len(pd_r2[feat_name]['values'][1]==(n_grid_1)))

  ### Plot ###
  feats = ['y', 'z']
  pd_p2 = get_pdp_and_ice(result['model'],
                          'regressor', 
                          df_train, 
                          features=feats, 
                          PDP=True, 
                          ICE=True, 
                          return_type='plot', 
                          grid_resolution=grid_resolution,
                          verbosity=0,
                          drop_invalid=False)
  
  for feat_name in feats:
    n_unique = len(df_train[feat_name].unique())
    n_grid = min(n_unique, grid_resolution)
    assert(pd_p2[feat_name].pd_results[0]['individual'].shape==(1, n_samples, n_grid))
    assert(pd_p2[feat_name].pd_results[0]['average'].shape==(1,n_grid))
    assert(pd_p2[feat_name].pd_results[0]['values'][0].shape==(n_grid,))

  # Check 2-way PDP
  feats = [('y', 'x')]
  pd_p2 = get_pdp_and_ice(result['model'],
                          'regressor', 
                          df_train, 
                          features=feats, 
                          PDP=True, 
                          ICE=False, 
                          return_type='plot', 
                          grid_resolution=grid_resolution,
                          verbosity=0,
                          drop_invalid=False)
  
  for feat_name in feats:
    n_unique_0 = len(df_train[feat_name[0]].unique())
    n_grid_0 = min(n_unique_0, grid_resolution)
    n_unique_1 = len(df_train[feat_name[1]].unique())
    n_grid_1 = min(n_unique_1, grid_resolution)
    assert(pd_p2[feat_name].pd_results[0]['average'].shape==(1, n_grid_0, n_grid_1))
    assert(len(pd_p2[feat_name].pd_results[0]['values'])==2)
    assert(len(pd_p2[feat_name].pd_results[0]['values'][0]==(n_grid_0)))
    assert(len(pd_p2[feat_name].pd_results[0]['values'][1]==(n_grid_1)))


def test_pdp_ice_classification(classification_task, tmp_path):
  """
  Check if PDP and ICE for classification tasks runs without errors, and that 
  the outputs are of the expected dimensions
  """
  
  df_train = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 3,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
                "z": ['a', 'b', 'b', 'c', 'a', 'c', 'a', 'b', 'a', 'c'] * 3,
                "t": [2, 2, 2, 2, 2, 3, 3, 3, 4, 4] * 3,
            }
        )
  n_samples = len(df_train)

  kwargs = dict()
  kwargs["drop_duplicates"] = False
  kwargs["hyperparameters"] = unittest_hyperparameters()
  kwargs["drop_unique"] = False
  kwargs["drop_useless_features"] = False

  result = classification_task.run(
      df=df_train,
      target="t",
      run_pdp=False,
      run_ice=False,
      model_directory=tmp_path,
      residuals_hyperparameters=unittest_hyperparameters(),
      presets="medium_quality_faster_train",
      **kwargs
  )
  
  grid_resolution = 10
  feats = ['y', 'z']

  ### Raw ###
  pd_r2 = get_pdp_and_ice(result['model'],
                          'classifier', 
                          df_train, 
                          features=feats, 
                          PDP=True, 
                          ICE=True, 
                          return_type='raw', 
                          grid_resolution=grid_resolution,
                          verbosity=0,
                          drop_invalid=False)
  
  n_unique_target = len(df_train['t'].unique())
  for feat_name in feats:
    n_unique = len(df_train[feat_name].unique())
    n_grid = min(n_unique, grid_resolution)
    assert(pd_r2[feat_name]['individual'].shape==(n_unique_target, n_samples, n_grid))
    assert(pd_r2[feat_name]['average'].shape==(n_unique_target, n_grid))
    assert(pd_r2[feat_name]['values'][0].shape==(n_grid,))
  

  # Check 2-way PDP
  feats = [('y', 'x')]
  pd_r2 = get_pdp_and_ice(result['model'],
                          'classifier', 
                          df_train, 
                          features=feats, 
                          PDP=True, 
                          ICE=True, 
                          return_type='raw', 
                          grid_resolution=grid_resolution,
                          verbosity=0,
                          drop_invalid=False)


  for feat_name in feats:
    n_unique_0 = len(df_train[feat_name[0]].unique())
    n_grid_0 = min(n_unique_0, grid_resolution)
    n_unique_1 = len(df_train[feat_name[1]].unique())
    n_grid_1 = min(n_unique_1, grid_resolution)
    assert(pd_r2[feat_name]['individual'].shape==(n_unique_target,n_samples, n_grid_0, n_grid_1))
    assert(pd_r2[feat_name]['average'].shape==(n_unique_target,n_grid_0, n_grid_1))
    assert(len(pd_r2[feat_name]['values'])==2)
    assert(len(pd_r2[feat_name]['values'][0]==(n_grid_0)))
    assert(len(pd_r2[feat_name]['values'][1]==(n_grid_1)))
  
  ### Plot ###
  feats = ['y', 'z']
  pd_p2 = get_pdp_and_ice(result['model'],
                          'classifier', 
                          df_train, 
                          features=feats, 
                          PDP=True, 
                          ICE=True, 
                          return_type='plot', 
                          grid_resolution=grid_resolution,
                          verbosity=0,
                          drop_invalid=False)
  
  for feat_name in feats:
    n_unique = len(df_train[feat_name].unique())
    n_grid = min(n_unique, grid_resolution)
    assert(pd_p2[feat_name].pd_results[0]['individual'].shape==(n_unique_target,n_samples,n_grid))
    assert(pd_p2[feat_name].pd_results[0]['average'].shape==(n_unique_target,n_grid))
    assert(pd_p2[feat_name].pd_results[0]['values'][0].shape==(n_grid,))

  # Check 2-way PDP
  feats = [('y', 'x')]
  pd_p2 = get_pdp_and_ice(result['model'],
                          'classifier', 
                          df_train, 
                          features=feats, 
                          PDP=True, 
                          ICE=False, 
                          return_type='plot', 
                          grid_resolution=grid_resolution,
                          verbosity=0,
                          drop_invalid=False)
  
  for feat_name in feats:
    n_unique_0 = len(df_train[feat_name[0]].unique())
    n_grid_0 = min(n_unique_0, grid_resolution)
    n_unique_1 = len(df_train[feat_name[1]].unique())
    n_grid_1 = min(n_unique_1, grid_resolution)
    assert(pd_p2[feat_name].pd_results[0]['average'].shape==(n_unique_target, n_grid_0, n_grid_1))
    assert(len(pd_p2[feat_name].pd_results[0]['values'])==2)
    assert(len(pd_p2[feat_name].pd_results[0]['values'][0]==(n_grid_0)))
    assert(len(pd_p2[feat_name].pd_results[0]['values'][1]==(n_grid_1)))
