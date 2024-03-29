import pandas as pd

from bciavm.pipelines.components.transformers import Transformer
from bciavm.pipelines.components.transformers.imputers import SimpleImputer
from bciavm import Boolean, Categorical, Double, Integer
from bciavm.utils import infer_feature_types

numeric_dtypes = [Integer, Double, Boolean]
categorical_dtypes = [Categorical]


class Imputer(Transformer):
    """Imputes missing data according to a specified imputation strategy."""
    name = "Imputer"
    hyperparameter_ranges = {
        "categorical_impute_strategy": ["most_frequent"],
        "numeric_impute_strategy": ["mean", "median", "most_frequent"]
    }
    _valid_categorical_impute_strategies = set(["most_frequent", "constant"])
    _valid_numeric_impute_strategies = set(["mean", "median", "most_frequent", "constant"])

    def __init__(self, categorical_impute_strategy="most_frequent",
                 categorical_fill_value=None,
                 numeric_impute_strategy="mean",
                 numeric_fill_value=None,
                 random_state=0, **kwargs):
        """Initalizes an transformer that imputes missing data according to the specified imputation strategy."

        Arguments:
            categorical_impute_strategy (string): Impute strategy to use for string, object, boolean, categorical dtypes. Valid values include "most_frequent" and "constant".
            numeric_impute_strategy (string): Impute strategy to use for numeric columns. Valid values include "mean", "median", "most_frequent", and "constant".
            categorical_fill_value (string): When categorical_impute_strategy == "constant", fill_value is used to replace missing data. The default value of None will fill with the string "missing_value".
            numeric_fill_value (int, float): When numeric_impute_strategy == "constant", fill_value is used to replace missing data. The default value of None will fill with 0.
        """
        if categorical_impute_strategy not in self._valid_categorical_impute_strategies:
            raise ValueError(f"{categorical_impute_strategy} is an invalid parameter. Valid categorical impute strategies are {', '.join(self._valid_numeric_impute_strategies)}")
        elif numeric_impute_strategy not in self._valid_numeric_impute_strategies:
            raise ValueError(f"{numeric_impute_strategy} is an invalid parameter. Valid impute strategies are {', '.join(self._valid_numeric_impute_strategies)}")

        parameters = {"categorical_impute_strategy": categorical_impute_strategy,
                      "numeric_impute_strategy": numeric_impute_strategy,
                      "categorical_fill_value": categorical_fill_value,
                      "numeric_fill_value": numeric_fill_value}
        parameters.update(kwargs)
        self._categorical_imputer = SimpleImputer(impute_strategy=categorical_impute_strategy,
                                                  fill_value=categorical_fill_value,
                                                  **kwargs)
        self._numeric_imputer = SimpleImputer(impute_strategy=numeric_impute_strategy,
                                              fill_value=numeric_fill_value,
                                              **kwargs)
        self._all_null_cols = None
        self._numeric_cols = None
        self._categorical_cols = None
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def fit(self, X, y=None):
        """Fits imputer to data. 'None' values are converted to np.nan before imputation and are
            treated as the same.

        Arguments:
            X (pd.DataFrame or np.array): The input training data of shape [n_samples, n_features]
            y (pd.Series, optional): The target training data of length [n_samples]

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self._all_null_cols = set(X.columns) - set(X.dropna(axis=1, how='all').columns)
        X_copy = X.copy()
        X_null_dropped = X_copy.drop(self._all_null_cols, axis=1, errors='ignore')

        numerics = [ 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        X_numerics = X_null_dropped.select_dtypes(include=numerics)
        if len(X_numerics.columns) > 0:
            self._numeric_imputer.fit(X_numerics, y)
            self._numeric_cols = X_numerics.columns

        categorical_dtypes = [ 'category', 'object', 'boolean' ]
        X_categorical = X_null_dropped.select_dtypes(include=categorical_dtypes)
        if len(X_categorical.columns) > 0:
            self._categorical_imputer.fit(X_categorical, y)
            self._categorical_cols = X_categorical.columns
        return self

    def transform(self, X, y=None):
        """Transforms data X by imputing missing values. 'None' values are converted to np.nan before imputation and are
            treated as the same.

        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Ignored.

        Returns:
            pd.DataFrame: Transformed X
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_null_dropped = X.copy()
        X_null_dropped.drop(self._all_null_cols, inplace=True, axis=1, errors='ignore')
        X_null_dropped.reset_index(inplace=True, drop=True)
        if X_null_dropped.empty:
            return X_null_dropped

        if 'unit_indx' not in X_null_dropped.columns:
            X_null_dropped = X_null_dropped.reset_index().rename({'index':'unit_indx'},axis=1)

        if self._numeric_cols is not None and len(self._numeric_cols) > 0:
            X_numeric = X_null_dropped[self._numeric_cols]
            X_null_dropped[X_numeric.columns] = self._numeric_imputer.transform(X_numeric)

        if self._categorical_cols is not None and len(self._categorical_cols) > 0:
            X_categorical = X_null_dropped[self._categorical_cols]
            X_null_dropped[X_categorical.columns] = self._categorical_imputer.transform(X_categorical)
        X_null_dropped = infer_feature_types(X_null_dropped)
        return X_null_dropped
