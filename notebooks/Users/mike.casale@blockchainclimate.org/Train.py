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
mlflow.set_experiment('/Users/mike.casale@blockchainclimate.org/Experiments/train')

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

import mlflow
mlflow.set_experiment('/Users/mike.casale@blockchainclimate.org/Experiments/tune')
best_run = mlflow.search_runs(order_by=['metrics.MdAPE ASC']).iloc[0]
print(f'MdAPE of Best Run: {best_run["metrics.MdAPE"]}')

# COMMAND ----------

best_run = best_run.reset_index()
best_run

# COMMAND ----------

params = {}
for col in best_run['index'].values:
  if 'params' in col:
    params[col]=best_run[best_run['index']==col][0].values[0]

# COMMAND ----------

params

# COMMAND ----------

from bciavm.pipelines import RegressionPipeline

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

#Uses the best params from the Tune-Hyperparameters Notebook
parameters = {'Imputer': {'categorical_impute_strategy': 'most_frequent',
                'numeric_impute_strategy': params['params.Imputer_numeric_impute_strategy'],
                'categorical_fill_value': None,
                'numeric_fill_value': None,
              },
             'One Hot Encoder': {'top_n': int(params['params.One Hot Encoder_top_n']),
                'features_to_encode': ['agg_cat'],
                'categories': None,
                'drop': None,
                'handle_unknown': 'ignore',
                'handle_missing': 'error',
              },
             'XGB Regressor': {'learning_rate': float(params['params.XGB Regressor_learning_rate']),
                        'max_depth': int(params['params.XGB Regressor_max_depth']),
                        'min_child_weight': int(params['params.XGB Regressor_min_child_weight']),
                        'reg_alpha': float(params['params.XGB Regressor_reg_alpha']),
                        'reg_lambda': float(params['params.XGB Regressor_reg_lambda']),
                        'n_estimators': int(params['params.XGB Regressor_n_estimators'])
             }
}


avm_pipeline = Pipeline(parameters=parameters)
avm_pipeline

# COMMAND ----------

max(y_train), min(y_train)

# COMMAND ----------

bciavm.__version__

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

input_example.dtypes

# COMMAND ----------

avm_pipeline.predict(input_example)

# COMMAND ----------

avm_pipeline.avm(input_example, batch_sample_sets=1, pred_interval=.9,  min_sample_sz=15)

# COMMAND ----------

class ConfModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.avm(X, pred_interval=.9,  min_sample_sz=15)

# COMMAND ----------

conf_model = ConfModel(avm_pipeline)
conf_model.predict(input_example)

# COMMAND ----------

mlflow.end_run()
mlflow.set_experiment('/Users/mike.casale@blockchainclimate.org/Experiments/train')
with mlflow.start_run(run_name='avm') as run:
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
    
    #log the training/testing data
    mlflow.log_artifact('/dbfs/FileStore/tables/avm/X_train.csv')
    mlflow.log_artifact('/dbfs/FileStore/tables/avm/X_test.csv')
    mlflow.log_artifact('/dbfs/FileStore/tables/avm/y_train.csv')
    mlflow.log_artifact('/dbfs/FileStore/tables/avm/y_test.csv')
    
    #log the raw model/pipeline artifact
    mlflow.log_artifact('/dbfs/FileStore/artifacts/avm_pipeline_'+str(bciavm.__version__)+'.pkl')
    
    #log the mlflow.sklearn flavor model with input example
    mlflow.sklearn.log_model(
                             avm_pipeline,
                             "model", 
                             input_example=input_example,
                            )
    


# COMMAND ----------

from mlflow.tracking import MlflowClient


mlflow.end_run()

model_name='avm'
model_version = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)

client = MlflowClient()

try:
  # Promote the new model version to Production
  client.transition_model_version_stage(
    name=model_name,
    version=int(model_version.version) - 1,
    stage="archived"
  )
except: pass

# Promote the new model version to Production
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Production"
)

# COMMAND ----------

# conf_model = Model(model)

mlflow.end_run()
with mlflow.start_run(run_name='avm-conf') as run:
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
    
    #log the training/testing data
    mlflow.log_artifact('/dbfs/FileStore/tables/avm/X_train.csv')
    mlflow.log_artifact('/dbfs/FileStore/tables/avm/X_test.csv')
    mlflow.log_artifact('/dbfs/FileStore/tables/avm/y_train.csv')
    mlflow.log_artifact('/dbfs/FileStore/tables/avm/y_test.csv')
    
    mlflow.log_params({'max_sample_sz':100, 'min_sample_sz':15, 'pred_interval':.9, 'conf_min':.5})
    
    #log the raw model/pipeline artifact
    mlflow.log_artifact('/dbfs/FileStore/artifacts/avm_pipeline_'+str(bciavm.__version__)+'.pkl')
    
    #log the mlflow.sklearn flavor model with input example
    mlflow.sklearn.log_model(
                             conf_model,
                             "model", 
                             input_example=input_example,
                            )
    



# COMMAND ----------

from mlflow.tracking import MlflowClient


mlflow.end_run()

model_name='avm-conf'
model_version = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)

client = MlflowClient()

try:
  # Promote the new model version to Production
  client.transition_model_version_stage(
    name=model_name,
    version=int(model_version.version) - 1,
    stage="archived"
  )
except: pass

# Promote the new model version to Production
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Production"
)

# COMMAND ----------

# dbutils.notebook.run("/Users/mike.casale@blockchainclimate.org/Batch Predict", 60)
