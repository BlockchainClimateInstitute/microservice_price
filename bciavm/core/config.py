from starlette.config import Config
from starlette.datastructures import Secret
import boto3, os
import pandas as pd

APP_VERSION = "0.0.1"
APP_NAME = "BCI AVM"
API_PREFIX = "/api"

TABLE_DIRECTORY = 'tables'
try: os.mkdir(TABLE_DIRECTORY)
except: pass

WGS84_a = 6378137.0
WGS84_b = 6356752.3

neighbourhood_radius = 0.1
box_in_km = neighbourhood_radius * 1.60934
half_side = 1000 * box_in_km

target = 'Price_p'

# config = Config(".env")
import dotenv
from dotenv import dotenv_values, load_dotenv

load_dotenv()

# config = dotenv_values(".env")
# print(config)
# API_KEY: Secret = config("API_KEY", cast=Secret)

ACCESS_KEY = os.environ["ACCESS_KEY"]
SECRET_KEY = os.environ["SECRET_KEY"]
BUCKET_NAME = os.environ["BUCKET_NAME"]

session = boto3.Session(aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
s3 = session.resource('s3')
your_bucket = s3.Bucket(BUCKET_NAME)
