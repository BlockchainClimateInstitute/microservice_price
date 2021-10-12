from sklearn.neighbors import KNeighborsRegressor as SKKNeighborsRegressor
from bciavm.model_family import ModelFamily
from bciavm.pipelines.components.estimators import Estimator
from bciavm.problem_types import ProblemTypes


class KNeighborsRegressor(Estimator):
    """KNeighborsRegressor."""
    name = "K Nearest Neighbors Regressor"
    hyperparameter_ranges = {}
    model_family = ModelFamily.LINEAR_MODEL
    supported_problem_types = [ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]

    def __init__(self,
                 n_neighbors=5,
                 weights='distance',
                 algorithm='auto',
                 p=2,
                 leaf_size=20,
                 metric='minkowski',
                 n_jobs=4,
                 random_seed=0,
                 **kwargs):

        parameters = {
            'n_neighbors': n_neighbors,
            'weights': weights,
            'algorithm': algorithm,
            'leaf_size': leaf_size,
            'p':p,
            'metric':metric,
            'n_jobs':n_jobs
        }

        parameters.update(kwargs)
        neighbors_regressor = SKKNeighborsRegressor(**parameters)
        super().__init__(parameters=parameters,
                         component_obj=neighbors_regressor,
                         random_seed=random_seed)

    def fit(self, X, y=None):
        X, y = super()._manage_woodwork(X, y)
        X = self._ensure_types(X)
        self.input_feature_names = list(X.columns)
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
        return super().predict(X)

    @property
    def feature_importance(self):
        return self._component_obj.coef_
