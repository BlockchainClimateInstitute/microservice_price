# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Predict
# MAGIC This notebook performs batch inference using the pre-trained, registered mlflow model. Use the **bci-avm-dask** cluster with the **Train** notebook
# MAGIC to see the model training.
# MAGIC 
# MAGIC 
# MAGIC <p align="center">
# MAGIC <img width=50% src="https://blockchainclimate.org/wp-content/uploads/2020/11/cropped-BCI_Logo_LR-400x333.png" alt="bciAVM" height="300"/>
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
# MAGIC             <img src="https://saturn-public-assets.s3.us-east-2.amazonaws.com/example-resources/dask.png" width="300">
# MAGIC         </td>
# MAGIC     </tr>
# MAGIC </table>
# MAGIC 
# MAGIC The Blockchain & Climate Institute (BCI) is a progressive think tank providing leading expertise in the deployment of emerging technologies for climate and sustainability actions. 
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

# MAGIC %md
# MAGIC <hr>
# MAGIC 
# MAGIC ## Environment Setup
# MAGIC 
# MAGIC The code in this notebook uses `bciavm` and `dask`.
# MAGIC 
# MAGIC In addition to the `bciavm` package, it relies on the following additional non-builtin libraries:
# MAGIC 
# MAGIC * [dask-ml](https://github.com/dask/dask-ml)

# COMMAND ----------

# DBTITLE 1,Import Libraries
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

# COMMAND ----------

# DBTITLE 1,Bring in entire EPC data (~15M rows)
spark.read.format("parquet").load("mnt/bct-transition-risk-data/epc_data/byLocation/DateRun_2021-02-07/{*}/domestic/certificates").createOrReplaceTempView("EPCData")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM EPCData

# COMMAND ----------

# DBTITLE 1,Load UK-Postcodes from freemaptools.com
spark.createDataFrame(pd.read_csv("/dbfs/FileStore/tables/ukpostcodes.csv")).createOrReplaceTempView("sqlPostcodeLonLat")
spark.createDataFrame(pd.read_csv("/dbfs/FileStore/tables/postcode_outcodes.csv")).createOrReplaceTempView("sqlOutcodeLonLat")

# COMMAND ----------

# DBTITLE 1,SQL Preprocessing
# MAGIC %sql
# MAGIC 
# MAGIC CREATE OR REPLACE TEMPORARY VIEW EPCDataFeatures AS
# MAGIC SELECT BUILDING_REFERENCE_NUMBER AS unit_indx
# MAGIC ,t1.POSTCODE
# MAGIC ,SPLIT(t1.POSTCODE, " ")[0] AS POSTCODE_OUTCODE
# MAGIC ,POSTTOWN AS POSTTOWN_e
# MAGIC ,PROPERTY_TYPE AS PROPERTY_TYPE_e
# MAGIC ,TOTAL_FLOOR_AREA AS TOTAL_FLOOR_AREA_e
# MAGIC ,NUMBER_HEATED_ROOMS AS NUMBER_HEATED_ROOMS_e
# MAGIC ,FLOOR_LEVEL AS FLOOR_LEVEL_e
# MAGIC ,CASE WHEN t3.latitude IS NOT NULL THEN t3.latitude ELSE t4.latitude END AS Latitude_m
# MAGIC ,CASE WHEN t3.longitude IS NOT NULL THEN t3.longitude ELSE t4.longitude END AS Longitude_m
# MAGIC ,CASE WHEN CAST (RIGHT(LEFT(t1.POSTCODE, 2), 1) AS INT) IS NULL THEN LEFT(t1.POSTCODE, 2) ELSE LEFT(t1.POSTCODE, 1) END AS POSTCODE_AREA
# MAGIC ,ROW_NUMBER() OVER (PARTITION BY BUILDING_REFERENCE_NUMBER ORDER BY INSPECTION_DATE DESC) AS rownum
# MAGIC FROM EPCData t1
# MAGIC LEFT JOIN sqlPostcodeLonLat t3 ON t1.POSTCODE = t3.Postcode
# MAGIC LEFT JOIN sqlOutcodeLonLat t4 ON SPLIT(t1.POSTCODE, " ")[0] = t4.postcode;
# MAGIC 
# MAGIC DROP TABLE IF EXISTS epcHomesToScore;
# MAGIC 
# MAGIC CREATE TABLE epcHomesToScore AS
# MAGIC SELECT unit_indx
# MAGIC ,POSTCODE
# MAGIC ,POSTCODE_OUTCODE
# MAGIC ,POSTTOWN_e
# MAGIC ,PROPERTY_TYPE_e
# MAGIC ,TOTAL_FLOOR_AREA_e
# MAGIC ,NUMBER_HEATED_ROOMS_e
# MAGIC ,FLOOR_LEVEL_e
# MAGIC ,Latitude_m
# MAGIC ,Longitude_m
# MAGIC ,POSTCODE_AREA
# MAGIC FROM EPCDataFeatures
# MAGIC WHERE rownum = 1;
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM epcHomesToScore

# COMMAND ----------

# DBTITLE 1,Convert to Pandas
data = spark.sql("SELECT * FROM epcHomesToScore").toPandas()

# COMMAND ----------

# DBTITLE 1,Set the MLFlow Experiment
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment('/Users/mike.casale@blockchainclimate.org/batch-predict')

# COMMAND ----------

# DBTITLE 1,Python Preprocessing
#TODO: merge w/ SQL preprocessing (above step)
data = bciavm.utils.bci_utils.preprocess_data(data.rename({'Postcode':'POSTCODE'},axis=1), drop_outlier=False, split_data=False)
data

# COMMAND ----------

# DBTITLE 1,Input Sample
input_example=data.dropna().sample(10)
input_example.dtypes

# COMMAND ----------

# DBTITLE 1,Load the AVM model from MLFlow
model_name='bci-test'
model_version='Production'

model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{model_version}"
    )   

model.predict(input_example)

# COMMAND ----------

# DBTITLE 1,Load the Confidence model from MLFlow
#NOTE: The actual model is exactly the same as above, but employs a custom model.avm(X) method via a wrapper which calls model.predict(X)

model_name='bci-conf-test'
model_version='Production'

model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{model_version}"
    )   

model.predict(input_example)

# COMMAND ----------

# DBTITLE 1,Dask Config
NPROCS = 2 #cpu cores per worker node
NWORKERS = 4 #number of workers
NPARTITIONS = NPROCS * NWORKERS #total parallel threads (assumes single threads (see the dask_cluster_init.sh bash start-up script))
print('NPARTITIONS = ', NPARTITIONS)

SAMPLE_ROWS = NPARTITIONS * 100

c = Client('127.0.0.1:8786')

print('waiting for workers...')
c.wait_for_workers(1)

print('done...')

# COMMAND ----------

# DBTITLE 1,Main Dask Prediction Logic

def mlflow_load_model(pred_type=None, model_name='bci-test', model_version='Production'):
    """Loads model from mlflow.

    Returns:
        mlflow.pyfunc loaded model
    """
    if pred_type == 'conf':
      model_name='bci-conf-test'
      model_version='Production'
      
    return mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{model_version}"
    )   
  
def get_unics():
  """Gets sample=1 for each unique combination of POSTCODE_AREA + PROPERTY_TYPE_e
     This is used to compute the confidence for all other properties which share 
     the POSTCODE_AREA + PROPERTY_TYPE_e combination
     
  Returns:
      pd.dataframe 
  """
  data['key'] = data['POSTCODE_AREA'] + data['PROPERTY_TYPE_e']
  unics = data['key'].unique()
  df = pd.DataFrame({})
  for u in unics:
    df = df.append(data[data['key']==u].sample(1))
  return df


def predict(X, pred_type, columns):
    """Main prediction logic

    Returns:
        pd.dataframe 
    """
    
    #loads the model from mlflow
    #the load model function is called for each dask partition in order to avoid the heavy lift of dask
    #distributing the model among workers (ie mlflow is better suited for this)
    try:
        model = mlflow_load_model(pred_type) 
    except:
        try:
            time.sleep(60)
            model = mlflow_load_model(pred_type)
        except:
            time.sleep(60)
            model = mlflow_load_model(pred_type)
    
    X['key'] = X['POSTCODE_AREA'] + X['PROPERTY_TYPE_e']
    
    unit_index = X['unit_indx'].values
    key = X['key'].values
    
    resp = pd.DataFrame(model.predict(X).values, columns=columns)
    resp['unit_indx'] = unit_index
    
    resp['key'] = key
    return resp



# COMMAND ----------

# DBTITLE 1,AVM Prediction
# MAGIC %%time
# MAGIC 
# MAGIC print('Building a dask dataframe...')
# MAGIC ddf = dd.from_pandas(data.sample(NPROCS*NWORKERS*100), npartitions=NPROCS*NWORKERS)
# MAGIC 
# MAGIC X_test_arr = dask.persist(ddf)
# MAGIC _ = wait(X_test_arr)
# MAGIC X_test_arr = X_test_arr[0]
# MAGIC 
# MAGIC print('Predicting...')
# MAGIC preds = X_test_arr.map_partitions(
# MAGIC         predict, pred_type=None, columns=['avm']
# MAGIC )
# MAGIC 
# MAGIC preds = preds.compute()
# MAGIC preds

# COMMAND ----------

# DBTITLE 1,Confidence Prediction
# MAGIC %%time
# MAGIC 
# MAGIC print('Building a dask dataframe...')
# MAGIC ddf = dd.from_pandas(_get_unics(), npartitions=NPROCS*NWORKERS)
# MAGIC 
# MAGIC X_test_arr = dask.persist(ddf)
# MAGIC _ = wait(X_test_arr)
# MAGIC X_test_arr = X_test_arr[0]
# MAGIC 
# MAGIC cols=['unit_id','avm','avm_lower','avm_upper','conf','ts','latest_production_version','latest_staging_version']
# MAGIC 
# MAGIC print('Predicting...')
# MAGIC confs = X_test_arr.map_partitions(
# MAGIC         predict, pred_type='conf', columns=cols
# MAGIC )
# MAGIC 
# MAGIC confs = confs.compute()
# MAGIC confs

# COMMAND ----------

from bciavm.utils.bci_utils import combine_confs

final_output = combine_confs(preds, confs)
final_output