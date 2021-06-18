import io
import math
from zipfile import ZipFile
from urllib.request import urlopen
import pandas as pd
import numpy as np
import bciavm
from bciavm.core.config import your_bucket, target, TABLE_DIRECTORY, WGS84_a, WGS84_b, box_in_km
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

target = 'Price_p'

def deg2rad(degrees):
    return math.pi * degrees / 180.0

def rad2deg(radians):
    return 180.0 * radians / math.pi

def get_latmin(lat):
    halfSide = 1000 * box_in_km

    # Radius of Earth at given latitude
    An = WGS84_a * WGS84_a * math.cos(lat)
    Bn = WGS84_b * WGS84_b * math.sin(lat)
    Ad = WGS84_a * math.cos(lat)
    Bd = WGS84_b * math.sin(lat)
    radius = math.sqrt((An * An + Bn * Bn) / (Ad * Ad + Bd * Bd))

    # Radius of the parallel at given latitude
    pradius = radius * math.cos(lat)
    latMin = lat - halfSide / radius
    latMax = lat + halfSide / radius
    return latMin

def get_latmax(lat):
    halfSide = 1000 * box_in_km

    # Radius of Earth at given latitude
    An = WGS84_a * WGS84_a * math.cos(lat)
    Bn = WGS84_b * WGS84_b * math.sin(lat)
    Ad = WGS84_a * math.cos(lat)
    Bd = WGS84_b * math.sin(lat)
    radius = math.sqrt((An * An + Bn * Bn) / (Ad * Ad + Bd * Bd))

    # Radius of the parallel at given latitude
    pradius = radius * math.cos(lat)
    latMin = lat - halfSide / radius
    latMax = lat + halfSide / radius
    return latMax


def get_pradius(lat):
    halfSide = 1000 * box_in_km

    # Radius of Earth at given latitude
    An = WGS84_a * WGS84_a * math.cos(lat)
    Bn = WGS84_b * WGS84_b * math.sin(lat)
    Ad = WGS84_a * math.cos(lat)
    Bd = WGS84_b * math.sin(lat)
    radius = math.sqrt((An * An + Bn * Bn) / (Ad * Ad + Bd * Bd))

    # Radius of the parallel at given latitude
    pradius = radius * math.cos(lat)
    return pradius

def get_postcodeOutcode_from_postcode(postcode):
    return postcode.split()[0]

def get_postcode_from_address(string):
    return string.split(' ')[-2] + ' ' + string.split(' ')[-1]

def get_postcodeArea_from_outcode(postcodeArea):
    if postcodeArea[1].isnumeric():
        return postcodeArea[0]
    else:
        return postcodeArea[0:2]

def ReadParquetFile(bucketName, fileLocation):
    df = pd.DataFrame()
    prefix_objs = your_bucket.objects.filter(Prefix=fileLocation)
    for s3_file in prefix_objs:
        obj = s3_file.get()
        df = df.append(pd.read_parquet(io.BytesIO(obj['Body'].read())))
    return df

def download_postcodes(path=TABLE_DIRECTORY + 'data/raw/ukpostcodes'):
    zipurl = 'https://www.freemaptools.com/download/full-postcodes/ukpostcodes.zip'

    # Download the file from the URL
    zipresp = urlopen(zipurl)

    # Create a new file on the hard drive
    tempzip = open("/tmp/tempfile.zip", "wb")

    # Write the contents of the downloaded file into the new file
    tempzip.write(zipresp.read())

    # Close the newly-created file
    tempzip.close()

    # Re-open the newly-created file with ZipFile()
    zf = ZipFile("/tmp/tempfile.zip")

    # Extract its contents into <extraction_path>
    # note that extractall will automatically create the path
    zf.extractall(path)

    # close the ZipFile instance
    zf.close()

def drop_outliers(data):
    upper_outliers, lower_outliers = np.quantile(data['Price_p'], 0.99), np.quantile(data['Price_p'], 0.01)
    data = data[(data['Price_p'] <= upper_outliers) & (data['Price_p'] >= lower_outliers)]
    data = data[
        (data['Price_p'] <= np.quantile(data['Price_p'], 0.99)
         ) & (data['Price_p'] >= 10000)]
    data = data[data['NUMBER_HABITABLE_ROOMS_e'].astype(float) < 10]
    data = data[data['FLAT_STOREY_COUNT_e'].astype(float).fillna(0) < 20]
    data = data[data['TOTAL_FLOOR_AREA_e'].astype(float) < 800]
    data = data[data['TOTAL_FLOOR_AREA_e'].astype(float) > 0]
    return data


def preprocess_data(data, drop_outlier=True, target='Price_p', test_size=0.15, split_data=True):

    if drop_outlier:
        data = drop_outliers(data)

    keep_cols = [x for x in data.columns if '_e' in x or x in ['Longitude_m', 'Latitude_m', 'Postcode']]
    if target in data.columns:
        keep_cols.append(target)

    try:
        match_types = ['1. Address Matched', '2. Address Matched No Spec', '3. No in Address Matched']
        data = data[data['TypeOfMatching_m'].isin(match_types)]
    except: pass

    if 'POSTCODE_AREA' in data.columns:
        keep_cols.append('POSTCODE')
        keep_cols.append('POSTCODE_OUTCODE')
        keep_cols.append('POSTCODE_AREA')

    data = data[[col for col in keep_cols if col in data.columns]]

    if 'POSTCODE_AREA' not in data.columns:
        try:
            data['POSTCODE'] = data['FULLADRESS_e'].apply(get_postcode_from_address)
            data['POSTCODE_OUTCODE'] = data['Postcode'].apply(get_postcodeOutcode_from_postcode)
            data['POSTCODE_AREA'] = data['POSTCODE_OUTCODE'].apply(get_postcodeArea_from_outcode)
        except: pass

    try:
        # drop outliers, convert floor level to integers
        data['Rooms'] = (data['NUMBER_HABITABLE_ROOMS_e'].astype(float) + data['NUMBER_HEATED_ROOMS_e'].astype(
            float)) / float(2)
    except:
        data['Rooms'] = data['NUMBER_HEATED_ROOMS_e'].astype(float)

    data['PROPERTY_TYPE_e'] = data['PROPERTY_TYPE_e'].astype(str).replace('nan', np.nan).fillna(
        'No PROPERTY_TYPE_e').astype(str)
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('NO DATA!', '0')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('NODATA!', '0')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('Basement', '-1')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('Ground', '0')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('1st', '1')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('2nd', '2')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('3rd', '3')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('4th', '4')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('5th', '5')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('6th', '6')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('9th', '9')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('11th', '11')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('12th', '12')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('13th', '13')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('14th', '14')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('15th', '15')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('16th', '16')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('17th', '17')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('18th', '18')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('19th', '19')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('20th', '20')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('21st or above', '21')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('top floor', '22')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('mid floor', '3')
    floor_levels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17','18', '19', '20', '21', '22', '-1']
    floor_levels = [x for x in data['FLOOR_LEVEL_e'].values if x not in floor_levels]
    data['FLOOR_LEVEL_e'] = np.where(data['FLOOR_LEVEL_e'].isin(floor_levels), np.nan, data['FLOOR_LEVEL_e']).astype(
        float)
    data['NUMBER_HEATED_ROOMS_e'] = data['NUMBER_HEATED_ROOMS_e'].astype(str).replace('nan', np.nan).fillna('0').astype(
        float)
    if target in data.columns:
        data = data[['POSTCODE',
                     'POSTCODE_OUTCODE',
                     'POSTCODE_AREA',
                     'POSTTOWN_e',
                     'PROPERTY_TYPE_e',
                     'TOTAL_FLOOR_AREA_e',
                     'NUMBER_HEATED_ROOMS_e',
                     'FLOOR_LEVEL_e',
                     'Latitude_m',
                     'Longitude_m',
                     target]]
    else:
        data = data[['POSTCODE',
                     'POSTCODE_OUTCODE',
                     'POSTCODE_AREA',
                     'POSTTOWN_e',
                     'PROPERTY_TYPE_e',
                     'TOTAL_FLOOR_AREA_e',
                     'NUMBER_HEATED_ROOMS_e',
                     'FLOOR_LEVEL_e',
                     'Latitude_m',
                     'Longitude_m']]

    if 'unit_indx' not in data.columns:
        data = data.reset_index()
        data = data.drop('index', axis=1)
        data = data.reset_index()
        data = data.rename({'index': 'unit_indx'}, axis=1)
        data = data.reset_index()
        data = data.drop('index', axis=1)

    data['TOTAL_FLOOR_AREA_e'] = data['TOTAL_FLOOR_AREA_e'].astype(float)
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].astype(float)
    data['NUMBER_HEATED_ROOMS_e'] = data['NUMBER_HEATED_ROOMS_e'].astype(float)

    if target in data.columns:
        data[target] = data[target].astype(float)

    if not split_data:
        return data

    X = data.drop('POSTCODE_AREA', axis=1)
    y = data['POSTCODE_AREA']
    X_train, X_holdout, y_train, y_holdout = bciavm.preprocessing.utils.split_data(X, y, problem_type='multiclass',
                                                                                   test_size=test_size)

    validate = X_holdout.reset_index().drop('index', axis=1)
    validate['POSTCODE_AREA'] = y_holdout.reset_index()['POSTCODE_AREA']

    train = X_train.reset_index().drop('index', axis=1)
    train['POSTCODE_AREA'] = y_train.reset_index()['POSTCODE_AREA']

    # additional processing to create train+test sets
    data = train
    data = data.dropna(subset=[target, 'TOTAL_FLOOR_AREA_e'])
    X, y = data, data[target]
    X, y = bciavm.preprocessing.utils.drop_nan_target_rows(X, y)

    # create a new feature upon which we will split the data into train+test sets
    X['key'] = X['POSTCODE_AREA'].astype(str) + '_' + X['PROPERTY_TYPE_e'].astype(str)
    vals = X['key'].value_counts().reset_index()
    vals = vals[vals['key'] > 5]
    X = X[X['key'].isin(list(vals['index'].values))]

    # split into train+test sets
    data_splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
    y = X[target]
    train, test = next(data_splitter.split(X, X['key']))
    X = X.drop('key', axis=1)
    X_train = X.iloc[train]
    X_test = X.iloc[test]
    y_train = y.iloc[train]
    y_test = y.iloc[test]
    X_train = X_train.drop(target, axis=1)
    X_test = X_test.drop(target, axis=1)
    return X_train, X_test, y_train, y_test
    

def merge_train_w_lookup_table(train, lookup_table):
    if 'PROPERTY_TYPE_e__' in lookup_table.columns:
        train = train.merge(lookup_table.rename({'PROPERTY_TYPE_e__': 'PROPERTY_TYPE_e'}, axis=1),
                            on=['POSTCODE', 'PROPERTY_TYPE_e'])
    else:
        train = train.merge(lookup_table, on=['POSTCODE', 'PROPERTY_TYPE_e'])

    new_X_train = pd.DataFrame({})

    for col in ['FLOOR_LEVEL_e', 'NUMBER_HEATED_ROOMS_e', 'TOTAL_FLOOR_AREA_e', 'Price_p']:
        agg_cols = [c for c in train.columns if col in c and c != col]
        for agg_col in agg_cols:
            if col != 'Price_p':
                new_X_train[col + '_minus_' + agg_col] = train[col] - train[agg_col]
            else:
                new_X_train[agg_col] = train[agg_col]

    new_X_train['Latitude_m'] = train['Latitude_m']
    new_X_train['Longitude_m'] = train['Longitude_m']
    new_X_train['PROPERTY_TYPE_e'] = train['PROPERTY_TYPE_e']
    new_X_train['POSTTOWN_e'] = train['POSTTOWN_e']
    new_X_train['POSTCODE'] = train['POSTCODE']
    new_X_train['POSTCODE_AREA'] = train['POSTCODE_AREA']
    new_X_train['density_count'] = train['density_count']
    new_X_train[target] = train[target]
    return new_X_train


def get_bounds(resp, y_predd):
    try:
        resp[ 'avm_lower' ] = round(
                resp[ 'avm' ].astype(float) + resp[ 'avm' ].astype(float) * resp[ 'avm_lower' ].astype(float),
                0)
    except:
        pass

    try:
        resp[ 'avm_upper' ] = round(
                resp[ 'avm' ].astype(float) + resp[ 'avm' ].astype(float) * resp[ 'avm_upper' ].astype(float),
                0)
    except:
        pass

    if (y_predd.values[0] - resp[ 'avm_lower' ].values[0] ) > (resp[ 'avm_upper' ].values[0] - y_predd.values[0]):
        fsd = y_predd.values[0] - resp[ 'avm_lower' ].values[0]
    else:
        fsd = resp[ 'avm_upper' ].values[0] - y_predd.values[0]

    conf = 1.0 - fsd / y_predd.values[0]
    resp['conf'] = [conf]

    return resp
  
  
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
    
    
    return predictions
  
  
def combine_confs(preds, confs):
    ll = []
    ct = 0
    for unit in preds['unit_indx'].values:
      try:
        resp = pd.DataFrame({})
        
        p = preds[preds['unit_indx']==unit]
        _key = p['key'].values[0]
        
        c = confs[confs['key']==_key]
        
        lower = 1.0 -  c['avm'] / c['avm_lower'] 
        upper = 1.0 -  c['avm'] / c['avm_upper']
        
        resp['unit_indx'] = p['unit_indx']
        resp['avm'] = p['avm']
        resp['avm_lower'] = [lower]
        resp['avm_upper'] = [upper]
        
        for _unit_id in resp['unit_indx'].unique():
          unit = resp[resp['unit_indx']==_unit_id]
          r = _get_bounds(unit, unit['avm'])
          ll.append(r)
          ct+=1
      except: pass
      
    return correct(pd.concat(ll))
      
