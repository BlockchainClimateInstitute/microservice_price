import pandas as pd
from sklearn.impute import SimpleImputer as SkImputer

from bciavm.pipelines.components.transformers import Transformer
from bciavm.utils import (
    _retain_custom_types_and_initalize_woodwork,
    infer_feature_types
)
from bciavm import Boolean
import woodwork as ww

class SimpleImputer(Transformer):
    """Imputes missing data according to a specified imputation strategy."""
    name = 'Simple Imputer'
    hyperparameter_ranges = {"impute_strategy": ["mean", "median", "most_frequent"]}

    def __init__(self, impute_strategy="most_frequent", fill_value=None, random_seed=0, **kwargs):
        """Initalizes an transformer that imputes missing data according to the specified imputation strategy."

        Arguments:
            impute_strategy (string): Impute strategy to use. Valid values include "mean", "median", "most_frequent", "constant" for
               numerical data, and "most_frequent", "constant" for object data types.
            fill_value (string): When impute_strategy == "constant", fill_value is used to replace missing data.
               Defaults to 0 when imputing numerical data and "missing_value" for strings or object data types.
            random_seed (int): Seed for the random number generator. Defaults to 0.
        """
        parameters = {"impute_strategy": impute_strategy,
                      "fill_value": fill_value}
        parameters.update(kwargs)
        imputer = SkImputer(strategy=impute_strategy,
                            fill_value=fill_value,
                            **kwargs)
        self._all_null_cols = None
        super().__init__(parameters=parameters,
                         component_obj=imputer,
                         random_seed=random_seed)

    def fit(self, X, y=None):
        """Fits imputer to data. 'None' values are converted to np.nan before imputation and are
            treated as the same.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): the input training data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, optional): the target training data of length [n_samples]

        Returns:
            self
        """
        X = infer_feature_types(X)

        # Convert all bool dtypes to category for fitting
        if (X.dtypes == bool).all():
            X = X.astype('category')

        self._component_obj.fit(X, y)
        self._all_null_cols = set(X.columns) - set(X.dropna(axis=1, how='all').columns)
        return self

    def transform(self, X, y=None):
        """Transforms input by imputing missing values. 'None' and np.nan values are treated as the same.

        Arguments:
            X (ww.DataTable, pd.DataFrame): Data to transform
            y (ww.DataColumn, pd.Series, optional): Ignored.

        Returns:
            ww.DataTable: Transformed X
        """
        X_ww = infer_feature_types(X)

        # Return early since bool dtype doesn't support nans and sklearn errors if all cols are bool
        if (X_ww.dtypes == bool).all():
            return X_ww

        X_null_dropped = X_ww.ww.drop(self._all_null_cols)
        X_t = self._component_obj.transform(X_ww)

        # Need this for test_simple_imputer_multitype_with_one_bool
        X_t = pd.DataFrame(X_t, columns=X_null_dropped.columns)
        if not X_null_dropped.empty:
            X_t.index = X_null_dropped.index

        return _retain_custom_types_and_initalize_woodwork(X_ww.ww.logical_types, X_t)

    def fit_transform(self, X, y=None):
        """Fits on X and transforms X

        Arguments:
            X (ww.DataTable, pd.DataFrame): Data to fit and transform
            y (ww.DataColumn, pd.Series, optional): Target data.

        Returns:
            ww.DataTable: Transformed X
        """
        return self.fit(X, y).transform(X, y)