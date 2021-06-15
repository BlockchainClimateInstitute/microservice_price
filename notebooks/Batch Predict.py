# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Predict
# MAGIC This notebook performs batch inference using the pre-trained, registered mlflow model. Use the **bci-avm-dask** cluster with the **Train** notebook
# MAGIC to see the model training.
# MAGIC 
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
# MAGIC             <img width=25% src="https://saturn-public-assets.s3.us-east-2.amazonaws.com/example-resources/dask.png" width="300">
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
# MAGIC In addition to the `bciavm` package, install the following additional non-builtin libraries:
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
from dask.distributed import wait
import shutil
import gc

# COMMAND ----------

# shutil.rmtree('/dbfs/FileStore/tables/avm_output/')
_date = str(datetime.now())
# os.mkdir('/dbfs/FileStore/tables/avm/avm_output_'+_date)

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
mlflow.set_experiment('/Users/mike.casale@blockchainclimate.org/Experiments/batch-predict')

# COMMAND ----------

# DBTITLE 1,Python Preprocessing
#TODO: merge w/ SQL preprocessing (above step)
data = bciavm.utils.bci_utils.preprocess_data(data.rename({'Postcode':'POSTCODE'},axis=1), 
                                              drop_outlier=False, 
                                              split_data=False)

data.to_csv('/dbfs/FileStore/tables/avm/epcPrice.csv')
data

# COMMAND ----------

# DBTITLE 1,Input Sample
input_example=data.dropna().sample(10)
input_example.dtypes

# COMMAND ----------

# DBTITLE 1,Dask Config
c = Client('127.0.0.1:8786')

print('waiting for workers...')
c.wait_for_workers(1)

print('done...')

# COMMAND ----------

def mlflow_load_model(pred_type=None, model_name='avm', model_version='Production'):
    """Loads model from mlflow.

    Returns:
        mlflow.pyfunc loaded model
    """
    if pred_type == 'conf':
      model_name='avm-conf'
      
    return mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{model_version}"
    )   
  
model = mlflow_load_model()
conf_model = mlflow_load_model(pred_type='conf')

# COMMAND ----------

model.predict(input_example)

# COMMAND ----------

conf_model.predict(input_example)

# COMMAND ----------

# DBTITLE 1,Main Batch Prediction Logic

def get_unics(data=None):
    """Gets sample=1 for each unique combination of POSTCODE_AREA + PROPERTY_TYPE_e
       This is used to compute the confidence for all other properties which share 
       the POSTCODE_AREA + PROPERTY_TYPE_e + NUMBER_HEATED_ROOMS_e + FLOOR_LEVEL_e combination

    Returns:
        pd.dataframe 
    """
    
    try: os.mkdir('/dbfs/FileStore/tables/avm/avm_conf/')
    except:pass
    
    
    if data is None:
        data = pd.read_csv('/dbfs/FileStore/tables/avm/epcPrice.csv')
    
    df = pd.DataFrame({})
    data['key'] = data['POSTCODE_AREA'] + data['PROPERTY_TYPE_e']
    unics = data['key'].unique()
    for u in unics:
      df = df.append(data[data['key']==u].sample(1))
    
    return df

def load_model():
    return bciavm.pipelines.RegressionPipeline.load('/dbfs/FileStore/artifacts/avm_pipeline_'+str(bciavm.__version__)+'.pkl')

def predict(X, model, columns=['avm']):
    """Main prediction logic
    Returns:
        pd.dataframe 
    """    
    X['key'] = X['POSTCODE_AREA'] + X['PROPERTY_TYPE_e'] 
    unit_index = X['unit_indx'].values
    key = X['key'].values
    resp = pd.DataFrame(model.predict(X).values, columns=columns)
    resp['unit_indx'] = unit_index
    resp['key'] = key
    del X
    gc.collect()
    return resp
    
def save(preds):
    filename='/dbfs/FileStore/tables/avm/avm_output/avm_output_'+str(datetime.now())+'.parquet.gzip'
    return preds.to_parquet(filename, compression='gzip')

def f(ct):
    for x in pd.read_csv('/dbfs/FileStore/tables/avm/epcPrice.csv', chunksize=500000):
        ct = ct + 1
        start_time = datetime.now()
        model = load_model()
        preds = predict(x, model)
        save(preds)
        end_time = datetime.now()
        del x
        del preds
        model = None
        gc.collect()
        print('Duration: {}'.format(end_time - start_time), ct)

    return ct
  
try: os.mkdir('/dbfs/FileStore/tables/avm/avm_output/')
except:pass

# COMMAND ----------

ct = 0
dask.compute(f(ct))

# COMMAND ----------

preds = dd.read_parquet('/dbfs/FileStore/tables/avm/avm_output/*.parquet.gzip', compression='gzip')
preds = preds.compute()
preds = preds.drop_duplicates('unit_indx')
preds

# COMMAND ----------

#Get all unique POSTCODE_AREA + PROPERTY_TYPE_e
try:
  unics = get_unics(data=data)
except:
  unics = get_unics()
unics

# COMMAND ----------

try: os.mkdir('/dbfs/FileStore/tables/avm/avm_conf')
except: pass

unics.to_parquet('/dbfs/FileStore/tables/avm/avm_conf/unics.parquet.gzip', compression='gzip')

# COMMAND ----------

unics = dd.read_parquet('/dbfs/FileStore/tables/avm/avm_conf/unics.parquet.gzip', compression='gzip')
unics = unics.compute()
unics

# COMMAND ----------

# DBTITLE 1,Confidence Prediction
# MAGIC %%time
# MAGIC 
# MAGIC print('Building a dask dataframe...')
# MAGIC ddf = dd.from_pandas(unics, npartitions=8)
# MAGIC X_test_arr = dask.persist(ddf)
# MAGIC _ = wait(X_test_arr)
# MAGIC X_test_arr = X_test_arr[0]
# MAGIC 
# MAGIC cols=['unit_id','avm','avm_lower','avm_upper','conf','ts','latest_production_version','latest_staging_version']
# MAGIC 
# MAGIC print('Predicting...')
# MAGIC confs = X_test_arr.map_partitions(
# MAGIC         predict, 
# MAGIC         model=conf_model,
# MAGIC         columns=cols
# MAGIC ).compute()
# MAGIC 
# MAGIC confs.to_parquet('/dbfs/FileStore/tables/avm/avm_conf/confs_output.parquet.gzip', compression='gzip')
# MAGIC confs

# COMMAND ----------

confs = dd.read_parquet('/dbfs/FileStore/tables/avm/avm_conf/confs_output.parquet.gzip', compression='gzip')
confs = confs.compute()
confs

# COMMAND ----------

def correct(predictions, conf_min=0.5):
    predictions[ 'avm' ] = round(predictions[ 'avm' ].astype(float), 0)
    predictions[ 'conf' ] = round(predictions[ 'conf' ].astype(float), 2)
    try :
      predictions[ 'avm' ] = np.where(predictions[ 'avm' ].astype(float) < 0.0, np.nan, predictions[ 'avm' ].astype(float))
    except :
      pass
    try :
      predictions[ 'avm_lower' ] = np.where(predictions[ 'avm_lower' ].astype(float) < 0.0, np.nan, predictions[ 'avm_lower' ].astype(float))
    except :
      pass
    try :
      predictions[ 'avm_upper' ] = np.where(predictions[ 'avm_upper' ].astype(float) < 0.0, np.nan, predictions[ 'avm_upper' ].astype(float))
    except :
      pass
    try :
      predictions[ 'avm_lower' ] = np.where(predictions[ 'avm_lower' ].astype(float) > predictions[ 'avm' ].astype(float), np.nan,
                                   predictions[ 'avm_lower' ].astype(float))
    except :
      pass
    try :
      predictions[ 'avm_upper' ] = np.where(predictions[ 'avm_upper' ].astype(float) < predictions[ 'avm' ].astype(float), np.nan,
                                   predictions[ 'avm_upper' ].astype(float))
    except :
      pass
    try :
      predictions.name = self.input_target_name
    except :
      pass

    try :
      predictions[ 'conf' ] = np.where(predictions[ 'conf' ].astype(float) < conf_min, '< 0.5',
                                   predictions[ 'conf' ].astype(float))
    except :
      pass
    
    predictions[ 'conf' ] = np.where(np.isnan(predictions[ 'avm_upper' ]), np.nan, predictions[ 'conf' ])
    predictions[ 'conf' ] = np.where(np.isnan(predictions[ 'avm_lower' ]), np.nan, predictions[ 'conf' ])
    return predictions

# COMMAND ----------

combined = preds.merge(confs.drop(['unit_indx', 'avm'],axis=1), on='key', how='left')
lower = combined['avm_lower'] / combined['avm'] - 1.0  
upper = combined['avm_upper'] / combined['avm'] - 1.0
combined['avm_upper'] = upper
combined['avm_lower'] = lower

combined[ 'avm_lower' ] = round(
            combined[ 'avm' ].astype(float) + combined[ 'avm' ].astype(float) * combined[ 'avm_lower' ].astype(float),
            0)
combined[ 'avm_upper' ] = round(
            combined[ 'avm' ].astype(float) + combined[ 'avm' ].astype(float) * combined[ 'avm_upper' ].astype(float),
            0)

combined['fsd'] = np.where((combined[ 'avm' ] - combined[ 'avm_lower' ]) >= (combined[ 'avm_upper' ] - combined[ 'avm' ]), combined[ 'avm' ] - combined[ 'avm_lower' ], combined[ 'avm_upper' ] - combined[ 'avm' ])

conf = 1.0 - combined['fsd'] / combined[ 'avm' ]
combined['conf'] = conf
combined = correct(combined, conf_min=0.5)
combined = combined.drop(['latest_staging_version', 'unit_id'], axis=1)
combined['conf'] = combined['conf'].fillna('< 0.5')
combined['avm_lower'] = np.where(np.isnan(combined['avm_lower']), combined['avm'] - combined['fsd'], combined['avm_lower'])
combined['avm_upper'] = np.where(np.isnan(combined['avm_upper']), combined['avm'] + combined['fsd'], combined['avm_upper'])
combined['avm_lower'] = np.where(combined['avm_lower'] < 0, 0.0, combined['avm_lower'])
combined = combined.drop('key',axis=1)
combined['avm_upper'] = round(combined['avm_upper'], 0)
combined['avm_lower'] = round(combined['avm_lower'], 0)
combined['fsd'] = round(combined['fsd'], 0)
combined

# COMMAND ----------

_date = str(datetime.now().date())
_date

# COMMAND ----------

combined.to_parquet('/dbfs/FileStore/tables/avm/final_output_'+_date+'.parquet.gzip', compression='gzip')

# COMMAND ----------

spark_df = spark.createDataFrame(combined)

spark_df.write.mode("overwrite").saveAsTable("/dbfs/FileStore/tables/avm_output_"+_date)