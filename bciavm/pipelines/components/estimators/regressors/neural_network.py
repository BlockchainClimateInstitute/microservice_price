from sklearn.neural_network import MLPRegressor as SKMLPRegressor
from bciavm.model_family import ModelFamily
from bciavm.pipelines.components.estimators import Estimator
from bciavm.problem_types import ProblemTypes


class MLPRegressor(Estimator):
    """MLPRegressor."""
    name = "MultiLayer Perceptron Regressor"
    hyperparameter_ranges = {}
    model_family = ModelFamily.LINEAR_MODEL
    supported_problem_types = [ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]

    def __init__(self,
                 activation='relu',
                 solver='adam',
                 alpha=0.043706006022706405,
                 batch_size='auto',
                 learning_rate='constant',
                 learning_rate_init=.001,
                 max_iter=500,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 n_iter_no_change=10,
                 random_seed=0,
                 early_stopping=True,
                 **kwargs):

        parameters = {
            'activation': activation,
            'solver': solver,
            'alpha': alpha,
            'batch_size': batch_size,
            'learning_rate':learning_rate,
            'learning_rate_init':learning_rate_init,
            'max_iter':max_iter,
            'early_stopping':early_stopping,
            'beta_1':beta_1,
            'beta_2':beta_2,
            'epsilon':epsilon,
            'n_iter_no_change':n_iter_no_change
        }

        parameters.update(kwargs)
        mlp_regressor = SKMLPRegressor(**parameters)
        super().__init__(parameters=parameters,
                         component_obj=mlp_regressor,
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
