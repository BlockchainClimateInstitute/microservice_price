from flask import Flask, request
from flask_restx import Api, Resource, fields
from collections import OrderedDict
import numpy as np
import pandas as pd
import joblib, pickle, json, requests, evalml, glob, os, urllib, sys
import evalml
from evalml.pipelines import RegressionPipeline
from evalml.preprocessing import load_data

sys.path.append('../')

app = Flask(__name__)
app.debug = True
api = Api(app, version='0.1', title='Price API - AVM',
    description='MicroService for automated valuation model',)

avm_columns = ['TOTAL_FLOOR_AREA','LONG','LAT','FLOOR_LEVEL','FLAT_TOP_STOREY','FLAT_STOREY_COUNT','PROPERTY_TYPE','NUMBER_HABITABLE_ROOMS','NUMBER_HEATED_ROOMS']
X, y = load_data('../data/processed/train.csv',index='POSTCODE',target='PRICE')
defaults = pd.read_csv('../data/processed/defaults.csv')
X = X[avm_columns]


# load the model from disk
modelfile = '../data/pipeline.pkl'
model = pickle.load(open(modelfile, "rb"))

# load the pipeline params from disk
with open('../data/params.json') as f:
  params = json.load(f)
    
# load the input_feature_names from disk
with open('../data/input_feature_names.json') as f:
  input_feature_names = json.load(f)

# in our case --> component_graph = ['Imputer', 'One Hot Encoder', 'XGBoost Regressor']
component_graph = [x for x in params]

address_res = api.model('Address', {
  'POSTCODE':fields.String(description='POSTCODE (default "EC4A 3EA")'),
	'AVM': fields.String(description='AVM Valuation')})

ns1 = api.namespace('AVM Service', description='ADDRESS operations')
@ns1.route('/<string:POSTCODE>')
@ns1.response(404, 'location not found')
@ns1.param('LONG', '-0.108378')
@ns1.param('LAT', '51.514902')
@ns1.param('FLOOR_LEVEL', '2nd')
@ns1.param('FLAT_TOP_STOREY', 'N')
@ns1.param('FLAT_STOREY_COUNT', '4.0')
@ns1.param('NUMBER_HEATED_ROOMS', '2.0')
@ns1.param('NUMBER_HABITABLE_ROOMS', '2.0')
@ns1.param('PROPERTY_TYPE', 'Flat')
@ns1.param('TOTAL_FLOOR_AREA', '28.72', default='28.72')

class POSTCODE(Resource):

    '''Show a single todo item and lets you delete them'''
    @ns1.doc('get_todo')
    @ns1.marshal_with(address_res)
    def get(self, POSTCODE):
      args = [i for i in request.args.keys()]
      print(args)
      # configure avm pipeline
      class AVMPipeline(RegressionPipeline):
        component_graph = component_graph
        custom_name = component_graph[2]
        input_feature_names = input_feature_names
        estimator = model

      avm = AVMPipeline(parameters=params, random_state=5).fit(X,y)
      dic = {'POSTCODE':POSTCODE}
      for col in avm_columns:
        if col in args: dic[col]=[request.args.get(col)]
        else: dic[col]=[defaults[col].values[0]]

      dfs = pd.DataFrame(dic)
      POSTCODE = dfs['POSTCODE'].values[0]
      dfs = dfs.set_index('POSTCODE')
      prediction = str(round(avm.predict(dfs)[0]))
      return {'POSTCODE':POSTCODE, 'AVM':prediction}

    
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port='5000', 
        )