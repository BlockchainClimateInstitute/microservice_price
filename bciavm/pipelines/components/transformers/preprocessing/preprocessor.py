
from bciavm.pipelines.components.transformers import Transformer
from bciavm.utils.bci_utils import preprocess_data
import pandas as pd

class PreprocessTransformer(Transformer) :
    """Transformer to Preprocessing"""
    name = "Preprocess Transformer"
    hyperparameter_ranges = {}

    def __init__(self, random_seed=0, target='Price_p', DATA_DICT = {
           "POSTCODE": {
              "Name": "POSTCODE",
              "PandasType": "object",
              "FillVal": 'None'
           },
            "POSTCODE_OUTCODE": {
                "Name": "POSTCODE_OUTCODE",
                "PandasType": "object",
                "FillVal": 'None'
            },
            "POSTTOWN_e": {
                "Name": "POSTTOWN_e",
                "PandasType": "object",
                "FillVal": 'None'
            },
            "PROPERTY_TYPE_e": {
                "Name": "PROPERTY_TYPE_e",
                "PandasType": "object",
                "FillVal": 'None'
            },
            "TOTAL_FLOOR_AREA_e": {
                "Name": "TOTAL_FLOOR_AREA_e",
                "PandasType": "float64",
                "FillVal": -1
            },
            "NUMBER_HEATED_ROOMS_e": {
                "Name": "NUMBER_HEATED_ROOMS_e",
                "PandasType": "object",
                "FillVal": -1
            },
            "FLOOR_LEVEL_e": {
                "Name": "FLOOR_LEVEL_e",
                "PandasType": "object",
                "FillVal": '0'
            },
            "Latitude_m": {
                "Name": "Latitude_m",
                "PandasType": "float64",
                "FillVal": -1.0
            },
            "Longitude_m": {
                "Name": "Longitude_m",
                "PandasType": "float64",
                "FillVal": -1.0
            },
            "Price_p": {
                "Name": "Price_p",
                "PandasType": "float64",
                "FillVal": -1.0
            },
            "POSTCODE_AREA": {
                "Name": "POSTCODE_AREA",
                "PandasType": "object",
                "FillVal": 'None'
            },
        }, **kwargs) :
        self.DTYPES = None
        self.columns = None
        self.final_features = None
        self.target = target

        self.lookup_table = pd.DataFrame({})
        for ct in range(1, 4):
            d = 'https://bciavm.s3.amazonaws.com/lookup_table_parquet/lookup_table'+str(ct)+'.parquet'
            self.lookup_table = self.lookup_table.append(pd.read_parquet(d))

        self.DATA_DICT = DATA_DICT

        parameters = {}
        parameters.update(kwargs)
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_seed=random_seed)
        

    def merge_tables(self, request_data):
        """ Merges the pre-computed lookup_table with the price dataset.

        Arguments:
            request_data: pd.DataFrame
        Returns:
            request_data: pd.DataFrame
        """
        lookup_table = self.lookup_table
        request_data = request_data.rename({'Postcode':'POSTCODE'}, axis=1)
        if 'POSTCODE' not in lookup_table.columns:
            self.lookup_table = pd.DataFrame({})
            for ct in range(1, 4):
                d = 'https://bciavm.s3.amazonaws.com/lookup_table_parquet/lookup_table' + str(ct) + '.parquet'
                self.lookup_table = self.lookup_table.append(pd.read_parquet(d))

        lookup_table = self.lookup_table

        if 'POSTCODE' not in request_data.columns:
            for col in request_data.columns:
                print(col)
                
        request_data = request_data.merge(lookup_table, on=['POSTCODE', 'PROPERTY_TYPE_e'], how='left')
        return request_data

    def _calc_aggregates(self, df, ops = ['mean', 'std', 'min', 'median', 'max'], agg_cols = ['TOTAL_FLOOR_AREA_e', 'NUMBER_HEATED_ROOMS_e', 'FLOOR_LEVEL_e']):
        """ Computes feature operations on aggregated POSTCODE features which provides the relative difference between
            individual properties and the aggregates within their POSTCODE.

        Arguments:
            df: pd.DataFrame
            ops (list): list of aggregation calculations includes in the pre-compute process
            agg_cols (list): list of columns to perform the operation on
        Returns:
            df: pd.DataFrame
        """

        for col in agg_cols:

            for op in ops:

                # calculate the difference between the aggregate and each individual unit
                try: df[col + '_' + 'minus' + '_' + op] = df[col].astype(float) - df[col + '__' + op].astype(float)
                except: df[col + '_' + 'minus' + '_' + op] = [np.nan for x in range(len(df))]
                
                # drop the aggregated features
                df = df.drop([col + '__' + op], axis=1)

        return df


    def fit(self, X, y=None) :
        return self

    def transform(self, X, y=None) :
        """Transforms data X by merging with precomputed values, and calculating new features.

        Arguments:
            X (ww.DataTable, pd.DataFrame): Data to transform
            y (ww.DataColumn, pd.Series, optional): Ignored.

        Returns:
            ww.DataTable: Transformed X
        """

        X_t = self.merge_tables(X)
        X_t = self._calc_aggregates(X_t)
        # X_t = X_t[[col for col in X_t.columns if 'POSTTOWN' not in col and 'POSTCODE' not in col and 'unit_indx' not in col]]
        X_t['agg_cat'] = X_t['POSTCODE_AREA'].astype(str) + '_' + X_t['PROPERTY_TYPE_e'].astype(str) + '_' + X_t['NUMBER_HEATED_ROOMS_e'].astype(str)
        return X_t



