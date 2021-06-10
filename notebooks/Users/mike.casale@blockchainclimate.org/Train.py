# Databricks notebook source
# MAGIC %md
# MAGIC # Train
# MAGIC This notebook performs model training.
# MAGIC To see the hyperparameter-tuning, use the **bci-avm-dask** cluster with the **Hypertuning** notebook.
# MAGIC 
# MAGIC <p align="center">
# MAGIC <img width=25% src="https://blockchainclimate.org/wp-content/uploads/2020/11/cropped-BCI_Logo_LR-400x333.png" alt="bciAVM" height="300"/>
# MAGIC </p>
# MAGIC 
# MAGIC [![PyPI](https://badge.fury.io/py/bciavm.svg?maxAge=2592000)](https://badge.fury.io/py/bciavm)
# MAGIC [![PyPI Stats](https://img.shields.io/badge/bciavm-avm-blue)](https://pypistats.org/packages/bciavm)
# MAGIC 
# MAGIC 
# MAGIC This notebook contains code to take a `mlflow` registered model and distribute its work with a `Dask` cluster. 
# MAGIC <table>
# MAGIC     <tr>
# MAGIC         <td>
# MAGIC             <img width=25% src="https://saturn-public-assets.s3.us-east-2.amazonaws.com/example-resources/dask.png" width="150">
# MAGIC         </td>
# MAGIC     </tr>
# MAGIC </table>
# MAGIC 
# MAGIC The [Blockchain Climate Institute](https://blockchainclimate.org) (BCI) is a progressive think tank providing leading expertise in the deployment of emerging technologies for climate and sustainability actions. 
# MAGIC 
# MAGIC As an international network of scientific and technological experts, BCI is at the forefront of innovative efforts, enabling technology transfers, to create a sustainable and clean global future.
# MAGIC 
# MAGIC # Automated Valuation Model (AVM) 
# MAGIC 
# MAGIC ### About
# MAGIC AVM is a term for a service that uses mathematical modeling combined with databases of existing properties and transactions to calculate real estate values. 
# MAGIC The majority of automated valuation models (AVMs) compare the values of similar properties at the same point in time. 
# MAGIC Many appraisers, and even Wall Street institutions, use this type of model to value residential properties. (see [What is an AVM](https://www.investopedia.com/terms/a/automated-valuation-model.asp) Investopedia.com)
# MAGIC 
# MAGIC For more detailed info about the AVM, please read the **About** paper found here `resources/2021-BCI-AVM-About.pdf`.
# MAGIC 
# MAGIC ### Valuation Process
# MAGIC <img src="resources/valuation_process.png" height="360" >
# MAGIC 
# MAGIC **Key Functionality**
# MAGIC 
# MAGIC * **Supervised algorithms** 
# MAGIC * **Tree-based & deep learning algorithms** 
# MAGIC * **Feature engineering derived from small clusters of similar properties** 
# MAGIC * **Ensemble (value blending) approaches** 
# MAGIC 
# MAGIC ### Set the required AWS Environment Variables
# MAGIC ```shell
# MAGIC export ACCESS_KEY=YOURACCESS_KEY
# MAGIC export SECRET_KEY=YOURSECRET_KEY
# MAGIC export BUCKET_NAME=bci-transition-risk-data
# MAGIC export TABLE_DIRECTORY=/dbfs/FileStore/tables/
# MAGIC ```
# MAGIC 
# MAGIC ### Next Steps
# MAGIC Read more about bciAVM on our [documentation page](https://blockchainclimate.org/thought-leadership/#blog):
# MAGIC 
# MAGIC ### How does it relate to BCI Risk Modeling?
# MAGIC <img src="resources/bci_flowchart_2.png" height="280" >
# MAGIC 
# MAGIC 
# MAGIC ### Technical & financial support for development provided by:
# MAGIC <a href="https://www.gcode.ai">
# MAGIC     <img width=15% src="https://staticfiles-img.s3.amazonaws.com/avm/gcode_logo.png" alt="GCODE.ai"  height="25"/>
# MAGIC </a>
# MAGIC 
# MAGIC 
# MAGIC ### Install [from PyPI](https://pypi.org/project/bciavm/)
# MAGIC ```shell
# MAGIC pip install bciavm
# MAGIC ```
# MAGIC 
# MAGIC This notebook covers the following steps:
# MAGIC - Import data from your local machine into the Databricks File System (DBFS)
# MAGIC - Download data from s3
# MAGIC - Train a machine learning models (or more technically, multiple models in a stacked pipeline) on the dataset
# MAGIC - Register the model in MLflow

# COMMAND ----------

import os
import time
import uuid
from datetime import datetime
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask
from dask.distributed import Client
import mlflow
from mlflow.tracking import MlflowClient
import bciavm
import re
from urllib.request import urlopen
import zipfile
from io import BytesIO
import io
from bciavm.core.config import your_bucket
from bciavm.utils.bci_utils import ReadParquetFile, get_postcodeOutcode_from_postcode, get_postcodeArea_from_outcode, drop_outliers, preprocess_data
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit


# COMMAND ----------

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment('/Users/mike.casale@blockchainclimate.org/train')

# COMMAND ----------

# MAGIC %md ## Importing Data
# MAGIC   
# MAGIC In this section, you download a dataset from the BCI s3 bucket and upload it to Databricks File System (DBFS).
# MAGIC 
# MAGIC 1. In addition to the s3 download, navigate to [https://www.freemaptools.com/download-uk-postcode-lat-lng.html/](https://www.freemaptools.com/download-uk-postcode-lat-lng.html/) and download both `ukpostcodes.csv` and `postcode_outcodes.csv` to your local machine.
# MAGIC 
# MAGIC 2. From this Databricks notebook, select *File* > *Upload Data*, and drag these files to the drag-and-drop target to upload them to the Databricks File System (DBFS). 
# MAGIC 
# MAGIC 3. Click *Next*. Some auto-generated code to load the data appears. Select *pandas*, and copy the example code. 
# MAGIC 
# MAGIC 4. Create a new cell, then paste in the sample code. It will look similar to the code shown in the following cell. Make these changes:

# COMMAND ----------

print(your_bucket)

# COMMAND ----------

dfPricesEpc = pd.DataFrame()
dfPrices = pd.DataFrame()

yearArray = ['2020', '2019']
for year in yearArray:

    #NOTE: To connect to s3, ensure your environment variables for AWS are set (see top of notebook for instructions)
    singlePriceEpcFile = pd.DataFrame(ReadParquetFile(your_bucket, 'epc_price_data/byDate/2021-02-04/parquet/' + year))
    dfPricesEpc = dfPricesEpc.append(singlePriceEpcFile)

dfPricesEpc['POSTCODE_OUTCODE'] = dfPricesEpc['Postcode'].apply(get_postcodeOutcode_from_postcode)
dfPricesEpc['POSTCODE_AREA'] = dfPricesEpc['POSTCODE_OUTCODE'].apply(get_postcodeArea_from_outcode)
dfPricesEpc.groupby('TypeOfMatching_m').count()['Postcode']
dfPricesEpc = dfPricesEpc.rename({'Postcode':'POSTCODE'},axis=1)

# COMMAND ----------

# MAGIC %md ## Preprocessing Data
# MAGIC Prior to training a model, check for missing values and split the data into training and validation sets.

# COMMAND ----------

X_train, X_test, y_train, y_test = bciavm.utils.bci_utils.preprocess_data(dfPricesEpc)

# COMMAND ----------

try: os.mkdir('/dbfs/FileStore/tables/avm')
except: pass

#NOTE: We need to log these to ensure our scoring process does not cheat
X_train.to_csv('/dbfs/FileStore/tables/avm/X_train.csv',index=False)
X_test.to_csv('/dbfs/FileStore/tables/avm/X_test.csv',index=False)
y_train.to_csv('/dbfs/FileStore/tables/avm/y_train.csv',index=False)
y_test.to_csv('/dbfs/FileStore/tables/avm/y_test.csv',index=False)

# COMMAND ----------

# X_train, X_test, y_train, y_test = pd.read_csv(cwd+'/data/X_train.csv'), pd.read_csv(cwd+'/data/X_test.csv'), pd.read_csv(cwd+'/data/y_train.csv'),  pd.read_csv(cwd+'/data/y_test.csv')

# COMMAND ----------

from bciavm.pipelines import RegressionPipeline

class Pipeline(RegressionPipeline):
    custom_name = 'AVM Pipeline'
    component_graph = {  'Preprocess Transformer': ['Preprocess Transformer'],
                         'Imputer': [
                             'Imputer', 
                             'Preprocess Transformer'
                         ],
                         'One Hot Encoder': [
                             'One Hot Encoder', 
                             'Imputer'
                         ],
                         'MultiLayer Perceptron Regressor': [
                             'MultiLayer Perceptron Regressor',
                             'One Hot Encoder'
                         ],
                         'K Nearest Neighbors Regressor': [
                             'K Nearest Neighbors Regressor',
                             'One Hot Encoder'
                         ],
                         'XGBoost Regressor': [
                             'XGBoost Regressor', 
                             'One Hot Encoder'
                         ],
                         'Node 1': [
                             'Linear Regressor',
                             'XGBoost Regressor',
                             'K Nearest Neighbors Regressor'
                         ],
                         'Node 2': [
                             'Linear Regressor',
                             'MultiLayer Perceptron Regressor',
                             'K Nearest Neighbors Regressor'
                         ],
                         'Node 3': [
                             'Linear Regressor',
                             'XGBoost Regressor',
                             'MultiLayer Perceptron Regressor'
                         ],
                         'Final Estimator': [
                             'Linear Regressor', 
                             'Node 1', 
                             'Node 2', 
                             'Node 3'
                         ]}
    
#Uses the best params from the Hypertuning Notebook
#TODO: automate by reading best params from mlflow logged trials
parameters = {'Imputer': {'categorical_impute_strategy': 'most_frequent',
              'numeric_impute_strategy': 'mean',
              'categorical_fill_value': None,
              'numeric_fill_value': None,
              },
             'One Hot Encoder': {'top_n': 6,
              'features_to_encode': ['agg_cat'],
              'categories': None,
              'drop': None,
              'handle_unknown': 'ignore',
              'handle_missing': 'error',
              },
             'MultiLayer Perceptron Regressor': {'activation': 'relu',
              'solver': 'adam',
              'alpha': 0.18448395702161716,
              'batch_size': 290,
              'learning_rate': 'constant',
              'learning_rate_init': 0.06010169396395971,
              'max_iter': 200,
              'early_stopping': True,
              'beta_1': 0.8,
              'beta_2': 0.999,
              'epsilon': 0.0001,
              'n_iter_no_change': 10
             },
             'K Nearest Neighbors Regressor': {'n_neighbors': 11,
              'weights': 'distance',
              'algorithm': 'auto',
              'leaf_size': 90,
              'p': 1,
              'metric': 'minkowski',
              'n_jobs': 4
             },
             'XGBoost Regressor': {'learning_rate': 0.06325261812661621,
                        'max_depth': 14,
                        'min_child_weight': 0.6718934260322275,
                        'reg_alpha': 0.043706006022706405,
                        'reg_lambda': 0.026408282583277758,
                        'n_estimators': 766
             },
             'Node 1': {'fit_intercept': True, 
                        'normalize': False, 
                        'n_jobs': -1
             },
             'Node 2': {'fit_intercept': True, 
                        'normalize': False, 
                        'n_jobs': -1
             },
             'Node 3': {'fit_intercept': True, 
                        'normalize': False, 
                        'n_jobs': -1
             },
             'Final Estimator': {'fit_intercept': True, 
                                 'normalize': False, 
                                 'n_jobs': -1
             }
}


avm_pipeline = Pipeline(parameters=parameters)
avm_pipeline

# COMMAND ----------

max(y_train), min(y_train)

# COMMAND ----------

avm_pipeline.fit(X_train, y_train)

# COMMAND ----------

bciavm.__version__

# COMMAND ----------

try: os.mkdir('/dbfs/FileStore/artifacts')
except: pass

avm_pipeline.save('/dbfs/FileStore/artifacts/avm_pipeline_'+str(bciavm.__version__)+'.pkl')
model = avm_pipeline

# COMMAND ----------

scores = model.score(X_test, y_test, objectives = ['MAPE',
                                                   'MdAPE',
                                                   'ExpVariance',
                                                   'MaxError',
                                                   'MedianAE',
                                                   'MSE',
                                                   'MAE',
                                                   'R2',
                                                   'Root Mean Squared Error'])
scores

# COMMAND ----------


input_example = X_test.dropna().sample(1)

# COMMAND ----------

# data = pd.read_csv('data/epcHomesToScore.csv')
# data = bciavm.utils.bci_utils.preprocess_data(data, drop_outlier=False, split_data=False)
# data

# COMMAND ----------

input_example.dtypes

# COMMAND ----------

avm_pipeline.predict(input_example)

# COMMAND ----------

class Model(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X, min_conf=0.50, )

# COMMAND ----------

model2 = Model(model)
model2.predict(input_example)

# COMMAND ----------

mlflow.end_run()
with mlflow.start_run(run_name='bci-test') as run:
    objectives = [ 'MAPE',
                   'MdAPE',
                   'ExpVariance',
                   'MaxError',
                   'MedianAE',
                   'MSE',
                   'MAE',
                   'R2',
                   'Root Mean Squared Error']
    
    for o in objectives:
        mlflow.log_metric(o, scores[o])
    
#     mlflow.log_artifact(cwd+'/data/X_train.csv')
#     mlflow.log_artifact(cwd+'/data/X_test.csv')
#     mlflow.log_artifact(cwd+'/data/y_train.csv')
#     mlflow.log_artifact(cwd+'/data/y_test.csv')
    
    mlflow.log_artifact('/dbfs/FileStore/artifacts/avm_pipeline_'+str(bciavm.__version__)+'.pkl')
    
    mlflow.sklearn.log_model(
                             model,
                             "model", 
                             input_example=input_example,
                            )
    


# COMMAND ----------

from mlflow.tracking import MlflowClient


mlflow.end_run()

model_name='bci-test'
model_version = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)

client = MlflowClient()

# Promote the new model version to Production
client.transition_model_version_stage(
  name=model_name,
  version=int(model_version.version) - 1,
  stage="archived"
)

# Promote the new model version to Production
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Production"
)

# COMMAND ----------

class Model(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.avm(X, 
                              batch_sample_sets=1, 
                              max_sample_sz=100, 
                              min_sample_sz=15, 
                              pred_interval=0.68, 
                              conf_min=0.5
                             )

# COMMAND ----------

conf_model = Model(model)

mlflow.end_run()
with mlflow.start_run(run_name='bci-test') as run:
    objectives = [ 'MAPE',
                   'MdAPE',
                   'ExpVariance',
                   'MaxError',
                   'MedianAE',
                   'MSE',
                   'MAE',
                   'R2',
                   'Root Mean Squared Error']
    
    for o in objectives:
        mlflow.log_metric(o, scores[o])
    
#     mlflow.log_artifact(cwd+'/data/X_train.csv')
#     mlflow.log_artifact(cwd+'/data/X_test.csv')
#     mlflow.log_artifact(cwd+'/data/y_train.csv')
#     mlflow.log_artifact(cwd+'/data/y_test.csv')
    
    mlflow.log_artifact('/dbfs/FileStore/artifacts/avm_pipeline_'+str(bciavm.__version__)+'.pkl')
    
    mlflow.sklearn.log_model(
                             conf_model,
                             "model", 
                             input_example=input_example,
                            )
    



# COMMAND ----------

from mlflow.tracking import MlflowClient


mlflow.end_run()

model_name='bci-conf-test'
model_version = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)

client = MlflowClient()

# Promote the new model version to Production
# client.transition_model_version_stage(
#   name=model_name,
#   version=int(model_version.version) - 1,
#   stage="archived"
# )

# Promote the new model version to Production
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Production"
)