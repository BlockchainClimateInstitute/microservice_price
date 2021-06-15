# Databricks notebook source
# MAGIC %md
# MAGIC # BCI AVM Hypertuning
# MAGIC ### Training the Machine Learning Models on Tabular Data: 
# MAGIC 
# MAGIC This notebook covers the following steps:
# MAGIC - Import data from AWS
# MAGIC - Visualize the data using Seaborn and matplotlib
# MAGIC - Run a parallel hyperparameter sweep to train machine learning models on the dataset
# MAGIC - Explore the results of the hyperparameter sweep with MLflow
# MAGIC - Register the best performing model in MLflow
# MAGIC 
# MAGIC ## Requirements
# MAGIC This notebook requires Databricks Runtime for Machine Learning or a similar spark/pyspark enabled environment setup locally (or elsewhere).  
# MAGIC If you are using Databricks Runtime 7.3 LTS ML or below, you must update the CloudPickle library using the commands in the following cell.

# COMMAND ----------

import cloudpickle

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing Data
# MAGIC   
# MAGIC In this section, you download a dataset from the web (AWS S3 Bucket).
# MAGIC 
# MAGIC 1. Ensure that you have installed `bciavm` using `pip install bciavm` in your local machine or on Databricks.

# COMMAND ----------

# DBTITLE 1,Add Libraries
import io
from bciavm.core.config import your_bucket
from bciavm.utils.bci_utils import ReadParquetFile, get_postcodeOutcode_from_postcode, get_postcodeArea_from_outcode, drop_outliers, preprocess_data
import pandas as pd
import bciavm
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
import sys
import hyperopt
from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from math import exp
import mlflow
import os
import gc
from bciavm.pipelines import RegressionPipeline
import numpy as np

# COMMAND ----------

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment('/Users/mike.casale@blockchainclimate.org/Experiments/tune-hyperparameters')

# COMMAND ----------

# DBTITLE 1,Get Data
dfPricesEpc = pd.DataFrame()
dfPrices = pd.DataFrame()

yearArray = ['2020', '2019']
for year in yearArray:
    singlePriceEpcFile = pd.DataFrame(ReadParquetFile(your_bucket, 'epc_price_data/byDate/2021-02-04/parquet/' + year))
    dfPricesEpc = dfPricesEpc.append(singlePriceEpcFile)

dfPricesEpc['POSTCODE_OUTCODE'] = dfPricesEpc['Postcode'].apply(get_postcodeOutcode_from_postcode)
dfPricesEpc['POSTCODE_AREA'] = dfPricesEpc['POSTCODE_OUTCODE'].apply(get_postcodeArea_from_outcode)
dfPricesEpc.groupby('TypeOfMatching_m').count()['Postcode']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessing Data
# MAGIC Prior to training a model, check for missing values and split the data into training and validation sets.

# COMMAND ----------

for n in range(100):
  try: 
    X_train, X_test, y_train, y_test = bciavm.utils.bci_utils.preprocess_data(
      dfPricesEpc.rename({'Postcode':'POSTCODE'},axis=1).sample(50000))
    break
  except: pass
  
X_train

# COMMAND ----------

# MAGIC %md
# MAGIC ##Hypertuning the AVM pipeline
# MAGIC 
# MAGIC The following code uses the `xgboost` and `scikit-learn` libraries to train a valuation model. It runs a parallel hyperparameter sweep to train multiple
# MAGIC models in parallel, using Hyperopt and SparkTrials. The code tracks the performance of each parameter configuration with MLflow.

# COMMAND ----------

# DBTITLE 1,Create a conda env log
version_info = sys.version_info
PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                              minor=version_info.minor,
                                              micro=version_info.micro)


conda_env = {'channels': ['defaults','conda-forge'],
            'dependencies': [
                'python={}'.format(PYTHON_VERSION),
                'pip',
                  {'pip': ['bciavm==1.21.5',
                           'dask-ml',
                           'hyperopt'
                          ],
                  },
            ],
            'name': 'mlflow-env'
}

# COMMAND ----------

try: os.mkdir('/dbfs/FileStore/tables/avm/tuning/')
except: pass

# COMMAND ----------

search_space = {
  'stacking': hp.choice('stacking', [True, True, False]),
  
  #Transformer tuning
  'numeric_impute_strategy': hp.choice('numeric_impute_strategy', ["mean", "median", "most_frequent"]),
  'top_n': hp.choice('top_n', [1, 2, 3, 4, 5, 6]),
    
  #K nearest neighbor tuning
  'n_neighbors': hp.choice('n_neighbors', [1, 2, 3, 4, 5, 6, 7, 8, 9]),
  'leaf_size': hp.choice('leaf_size', [1, 2, 3, 4, 5, 6, 7]),
  'p': 2,
    
  #MultiLayer Perceptron Regressor tuning
  'activation': 'relu',
  'solver': 'adam',
  'batch_size': scope.int(hp.quniform('batch_size', 600, 700, 1)),
  'alpha': hp.choice('alpha', [0.02, 0.03]),
  'learning_rate_init': hp.choice('learning_rate_init', [0.005, 0.01, 0.1, 0.125, 0.133, 0.15]),
  'max_iter': hp.choice('max_iter', [150, 160, 170, 180, 190, 200]),
  'beta_1': hp.choice('beta_1', [0.2, 0.25, 0.3]),
  'epsilon': hp.choice('epsilon', [1e-08, 1e-07, 1e-09]),

  #XGBoost Regressor tuning
  'n_estimators': scope.int(hp.quniform('n_estimators', 400, 600, 10)),
  'max_depth': scope.int(hp.quniform('max_depth', 80, 200, 5)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
  'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
  'min_child_weight': hp.choice('min_child_weight', [5, 6, 7]),
  
  #Final Estimator XGBoost Regressor tuning
  'fe_n_estimators': scope.int(hp.quniform('fe_n_estimators', 100, 1000, 10)),
  'fe_max_depth': scope.int(hp.quniform('fe_max_depth', 4, 100, 1)),
  'fe_learning_rate': hp.loguniform('fe_learning_rate', -3, 0),
  'fe_reg_alpha': hp.choice('fe_reg_alpha', [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]),
  'fe_reg_lambda': hp.choice('fe_reg_lambda', [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]),
  'fe_min_child_weight': hp.choice('fe_min_child_weight', [1, 2]),
  'fe_metric':'mae',
  'fe_objective': 'reg:squarederror',
  'fe_seed': 123, # Set a seed for deterministic training
}

# COMMAND ----------

def train_model(params):

    mlflow.set_experiment('/Users/mike.casale@blockchainclimate.org/Experiments/tune')
    
    # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
    mlflow.sklearn.autolog()

    with mlflow.start_run(nested=True):
      
        if params['stacking']:
            class Pipeline(RegressionPipeline):
                custom_name = 'AVM Pipeline'
                component_graph = {  'Preprocess Transformer': [
                                         'Preprocess Transformer'
                                     ],
                                     'Imputer': [
                                         'Imputer', 
                                         'Preprocess Transformer'
                                     ],
                                     'One Hot Encoder': [
                                         'One Hot Encoder', 
                                         'Imputer'
                                     ],
                                     'K Nearest Neighbors': [
                                         'K Nearest Neighbors Regressor',
                                         'One Hot Encoder'
                                     ],
                                     'XGB Regressor': [
                                         'XGBoost Regressor', 
                                         'One Hot Encoder'
                                     ],
                                     'ML Perceptron Regressor': [
                                         'MultiLayer Perceptron Regressor', 
                                         'One Hot Encoder'
                                     ],
                                     'Linear Regressor Stack': [
                                         'Linear Regressor',
                                         'K Nearest Neighbors',
                                         'XGB Regressor'
                                     ],
                                     'Final Estimator': [
                                         'XGBoost Regressor', 
                                         'Linear Regressor Stack',
                                         'K Nearest Neighbors',
                                         'XGB Regressor',
                                         'ML Perceptron Regressor',
                                         'One Hot Encoder'
                                     ]}

            #Uses the best params from the Hypertuning Notebook
            #TODO: automate by reading best params from mlflow logged trials
            parameters = {'Imputer': {'categorical_impute_strategy': 'most_frequent',
                            'numeric_impute_strategy': params['numeric_impute_strategy'],
                            'categorical_fill_value': None,
                            'numeric_fill_value': None,
                          },
                         'One Hot Encoder': {'top_n': params['top_n'],
                            'features_to_encode': ['agg_cat'],
                            'categories': None,
                            'drop': None,
                            'handle_unknown': 'ignore',
                            'handle_missing': 'error',
                          },
                         'K Nearest Neighbors': {'n_neighbors': params['n_neighbors'],
                            'weights': 'distance',
                            'algorithm': 'auto',
                            'leaf_size': params['leaf_size'],
                            'p': params['p'],
                            'metric': 'minkowski',
                            'n_jobs': 1
                         },
                         'ML Perceptron Regressor': {'activation': 'relu',
                            'solver': 'adam',
                            'alpha': float(params['alpha']),
                            'batch_size': int(params['batch_size']),
                            'learning_rate': 'constant',
                            'learning_rate_init': float(params['learning_rate_init']),
                            'max_iter': int(params['max_iter']),
                            'early_stopping': True,
                            'beta_1': float(params['beta_1']),
                            'beta_2': 0.999,
                            'epsilon': float(params['epsilon']),
                            'n_iter_no_change': 10
                         },
                         'XGB Regressor': {'learning_rate': params['learning_rate'],
                                    'max_depth': params['max_depth'],
                                    'min_child_weight': params['min_child_weight'],
                                    'reg_alpha': params['reg_alpha'],
                                    'reg_lambda': params['reg_lambda'],
                                    'n_estimators': params['n_estimators']
                         },
                         'Final Estimator': {'learning_rate': params['fe_learning_rate'],
                                            'max_depth': params['fe_max_depth'],
                                            'min_child_weight': params['fe_min_child_weight'],
                                            'reg_alpha': params['fe_reg_alpha'],
                                            'reg_lambda': params['fe_reg_lambda'],
                                            'n_estimators': params['fe_n_estimators']
                         }
            }
        
        else:
            class Pipeline(RegressionPipeline):
                custom_name = 'AVM Pipeline'
                component_graph = {  'Preprocess Transformer': [
                                         'Preprocess Transformer'
                                     ],
                                     'Imputer': [
                                         'Imputer', 
                                         'Preprocess Transformer'
                                     ],
                                     'One Hot Encoder': [
                                         'One Hot Encoder', 
                                         'Imputer'
                                     ],
                                     'XGB Regressor': [
                                         'XGBoost Regressor', 
                                         'One Hot Encoder'
                                     ]}

            #Uses the best params from the Hypertuning Notebook
            #TODO: automate by reading best params from mlflow logged trials
            parameters = {'Imputer': {'categorical_impute_strategy': 'most_frequent',
                            'numeric_impute_strategy': params['numeric_impute_strategy'],
                            'categorical_fill_value': None,
                            'numeric_fill_value': None,
                          },
                         'One Hot Encoder': {'top_n': params['top_n'],
                            'features_to_encode': ['agg_cat'],
                            'categories': None,
                            'drop': None,
                            'handle_unknown': 'ignore',
                            'handle_missing': 'error',
                          },
                         'XGB Regressor': {'learning_rate': params['fe_learning_rate'],
                                    'max_depth': params['fe_max_depth'],
                                    'min_child_weight': params['fe_min_child_weight'],
                                    'reg_alpha': params['fe_reg_alpha'],
                                    'reg_lambda': params['fe_reg_lambda'],
                                    'n_estimators': params['fe_n_estimators']
                         }
            }
            
        for component in parameters:
          for param in parameters[component]:
            _component_param = component + '_' + param
            mlflow.log_param(_component_param, parameters[component][param])

        avm_pipeline = Pipeline(parameters=parameters)
        avm_pipeline.fit(X_train, y_train)

        # Compute and return trial error
        scores = avm_pipeline.score(X_test, 
                                         y_test, 
                                         objectives=['MAPE',
                                                   'MdAPE',
                                                   'ExpVariance',
                                                   'MaxError',
                                                   'MedianAE',
                                                   'MSE',
                                                   'MAE',
                                                   'R2',
                                                   'Root Mean Squared Error'])
        MdAPE = scores['MdAPE']

        #Examine the learned feature importances output by the model as a sanity-check.
        fi = pd.DataFrame({'feature':avm_pipeline.get_component("XGB Regressor").input_feature_names,'importance':avm_pipeline.get_component("XGB Regressor").feature_importance}).sort_values(by='importance', ascending=False)

        #Log the feature importances output as an artifact
        artifact_path = '/dbfs/FileStore/tables/avm/tuning/XGBoost_importance.csv'
        fi.to_csv(artifact_path,index=False)
        mlflow.log_artifact(artifact_path)

        #Log the scoring metrics
        mlflow.log_metric('MAPE', scores['MAPE'])
        mlflow.log_metric('MdAPE', scores['MdAPE'])
        mlflow.log_metric('ExpVariance', scores['ExpVariance'])
        mlflow.log_metric('MaxError', scores['MaxError'])
        mlflow.log_metric('MedianAE', scores['MedianAE'])
        mlflow.log_metric('MSE', scores['MSE'])
        mlflow.log_metric('MAE', scores['MAE'])
        mlflow.log_metric('R2', scores['R2'])
        mlflow.log_metric('Root Mean Squared Error', scores['Root Mean Squared Error'])

        #Log an input example for future reference
        input_example = X_train.dropna().sample(1)

        #Log the mlflow model, along with the conda environment and input example
        mlflow.sklearn.log_model(
                             avm_pipeline,
                             "avm", 
                             conda_env=conda_env,
                             input_example=input_example
                            )

        # fmin minimizes the MdAPE (median absolute percentage error)
        return {'status': STATUS_OK, 'loss': scores['MdAPE']}

# Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep. 
# A reasonable value for parallelism is the square root of max_evals.
spark_trials = SparkTrials(parallelism=3)

# Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
mlflow.set_experiment('/Users/mike.casale@blockchainclimate.org/Experiments/tune')
with mlflow.start_run(run_name='avm_tune'):
    best_params = fmin(
        fn=train_model, 
        space=search_space, 
        algo=tpe.suggest, 
        #max_evals=42,
        timeout=10800,
        trials=spark_trials,
        rstate=np.random.RandomState(123)
        )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Use MLflow to view the results
# MAGIC Open up the Experiment Runs sidebar to see the MLflow runs. Click on Date next to the down arrow to display a menu, and select 'MdAPE' to display the runs sorted by the MdAPE metric. The lowest MdAPE value is ~10%. 
# MAGIC 
# MAGIC MLflow tracks the parameters and performance metrics of each run. Click the External Link icon <img src="https://docs.databricks.com/_static/images/external-link.png"/> at the top of the Experiment Runs sidebar to navigate to the MLflow Runs Table.

# COMMAND ----------

# MAGIC %md
# MAGIC Now investigate how the hyperparameter choice correlates with MdAPE. Click the "+" icon to expand the parent run, then select all runs except the parent, and click "Compare". Select the Parallel Coordinates Plot.
# MAGIC 
# MAGIC The Parallel Coordinates Plot is useful in understanding the impact of parameters on a metric. You can drag the pink slider bar at the upper right corner of the plot to highlight a subset of MdAPE values and the corresponding parameter values. The plot below highlights the highest MdAPE values:
# MAGIC 
# MAGIC <img src="https://docs.databricks.com/_static/images/mlflow/end-to-end-example/parallel-coordinates-plot.png"/>
# MAGIC 
# MAGIC Notice that all of the top performing runs have a low value for reg_lambda and learning_rate. 
# MAGIC 
# MAGIC You could run another hyperparameter sweep to explore even lower values for these parameters. For simplicity, that step is not included here.

# COMMAND ----------

# MAGIC %md
# MAGIC You used MLflow to log the model produced by each hyperparameter configuration. The following code finds the best performing run and saves the model to the model registry.

# COMMAND ----------

best_run = mlflow.search_runs(order_by=['metrics.MdAPE ASC']).iloc[0]
print(f'MdAPE of Best Run: {best_run["metrics.MdAPE"]}')