import mlflow
import mlflow.pyfunc

# wrap the model in order to call model.avm(X)
# which enables ParallelPostFit to be used with the predict() method
class AVMWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        # the predict() method wraps the avm() method which is VERY slow and why we want
        # to use DASK to speed up
        return self.model.avm(X).values

