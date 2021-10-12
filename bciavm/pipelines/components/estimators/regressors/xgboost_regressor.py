from skopt.space import Integer, Real

from bciavm.model_family import ModelFamily
from bciavm.pipelines.components.estimators import Estimator
from bciavm.problem_types import ProblemTypes
from bciavm.utils.gen_utils import (
    _rename_column_names_to_numeric,
    import_or_raise
)

import time, sys, random, json
import hyperopt
from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from math import exp
from dask_ml.wrappers import ParallelPostFit


class XGBoostRegressor(Estimator):
    """XGBoost Regressor."""
    name = "XGBoost Regressor"
    hyperparameter_ranges = {
        "eta": Real(0.000001, 1),
        "max_depth": Integer(1, 20),
        "min_child_weight": Real(1, 10),
        "n_estimators": Integer(1, 1000),
    }

    search_space = {
        'max_depth' : scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate' : hp.loguniform('learning_rate', -3, 0),
        'reg_alpha' : hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda' : hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight' : hp.loguniform('min_child_weight', -1, 3),
        'n_estimators' : hp.choice('n_estimators', [ x for x in range(100, 1000) ]),
        'objective' : 'reg:squarederror',
        'metric' : 'mae',
        'seed' : 123,  # Set a seed for deterministic training
    }

    params = {
        'metric': 'mae',
        'objective': 'reg:squarederror',
        'verbose_eval': True,
        'seed': 123,
        'reg_alpha': 0.043706006022706405,
        'max_depth': 14,
        'learning_rate': 0.06325261812661621,
        'min_child_weight': 0.6718934260322275,
        'reg_lambda': 0.026408282583277758
    }
    model_family = ModelFamily.XGBOOST
    supported_problem_types = [ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]

    # xgboost supports seeds from -2**31 to 2**31 - 1 inclusive. these limits ensure the random seed generated below
    # is within that range.
    SEED_MIN = -2**31
    SEED_MAX = 2**31 - 1

    def __init__(self, learning_rate=0.06325261812661621,
                        max_depth=14,
                        min_child_weight=0.6718934260322275,
                        n_estimators=766,
                        reg_alpha=0.043706006022706405,
                        reg_lambda=0.026408282583277758,
                        random_seed=0,
                        **kwargs):

        parameters = {"learning_rate": learning_rate,
                      "max_depth": max_depth,
                      "min_child_weight": min_child_weight,
                      "reg_alpha": reg_alpha,
                      "reg_lambda": reg_lambda,
                      "n_estimators": n_estimators}

        parameters.update(kwargs)

        xgb_error_msg = "XGBoost is not installed. Please install using `pip install xgboost.`"
        xgb = import_or_raise("xgboost", error_msg=xgb_error_msg)
        xgb_Regressor = xgb.XGBRegressor(n_jobs=4, random_state=random_seed,
                                         **parameters)
        super().__init__(parameters=parameters,
                         component_obj=xgb_Regressor,
                         random_seed=random_seed)

    def fit(self, X, y=None):
        X, y = super()._manage_woodwork(X, y)
        X = self._ensure_types(X)
        self.input_feature_names = list(X.columns)
        X = _rename_column_names_to_numeric(X, flatten_tuples=False)
        self._component_obj.fit(X, y)
        return self

    def _ensure_types(self, X):
        for col in X.columns:
            try: X[col] = X[col].astype(float)
            except: X = X.drop(col, axis=1)
            if col in ['unit_indx','_c0']:
                try: X = X.drop(col, axis=1)
                except: pass
        return X

    def predict(self, X):
        X = self._ensure_types(X)
        X = X[self.input_feature_names]
        X = _rename_column_names_to_numeric(X, flatten_tuples=False)
        return super().predict(X)

    @property
    def feature_importance(self):
        return self._component_obj.feature_importances_
