import copy
import inspect
import io
import os
import re
import sys
import traceback
from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy as np
import cloudpickle
import pandas as pd
from dask_ml.wrappers import ParallelPostFit
from .dask_wrapper import AVMWrapper

from bciavm.utils.bci_utils import preprocess_data
from .components import (
    PCA,
    DFSTransformer,
    Estimator,
    LinearDiscriminantAnalysis,
    StackedEnsembleRegressor
)
from .components.utils import all_components, handle_component_class
from datetime import datetime
from bciavm.exceptions import (
    IllFormattedClassNameError,
    ObjectiveCreationError,
    PipelineScoreError
)
from bciavm.objectives import get_objective
from bciavm.pipelines import ComponentGraph
from bciavm.pipelines.pipeline_meta import PipelineBaseMeta
from bciavm.problem_types import is_binary
from pathlib import Path
from bciavm.utils import (
    classproperty,
    get_logger,
    import_or_raise,
    infer_feature_types,
    jupyter_check,
    log_subtitle,
    log_title,
    safe_repr
)
import mlflow
from mlflow.tracking import MlflowClient
from ..utils.gen_utils import fill_nulls
import pkgutil
from joblib import Parallel, delayed

client = MlflowClient()

target = 'Price_p'

logger = get_logger(__file__)


class PipelineBase(ABC, metaclass=PipelineBaseMeta):
    """Base class for all pipelines."""

    @property
    @classmethod
    @abstractmethod
    def component_graph(cls):
        """Returns list or dictionary of components representing pipeline graph structure

        Returns:
            list(str / ComponentBase subclass): List of ComponentBase subclasses or strings denotes graph structure of this pipeline
        """

    custom_hyperparameters = None
    custom_name = None
    problem_type = None

    def __init__(self, parameters, random_seed=0, DATA_DICT = {
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
        },
        target='Price_p'):
        """Machine learning pipeline made out of transformers and a estimator.

        Required Class Variables:
            component_graph (list): List of components in order. Accepts strings or ComponentBase subclasses in the list

        Arguments:
            parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
                 An empty dictionary {} implies using all default values for component parameters.
            random_seed (int): Seed for the random number generator. Defaults to 0.
        """
        self.random_seed = random_seed
        if isinstance(self.component_graph, list):  # Backwards compatibility
            self._component_graph = ComponentGraph().from_list(self.component_graph, random_seed=self.random_seed)
        else:
            self._component_graph = ComponentGraph(component_dict=self.component_graph, random_seed=self.random_seed)
        self._component_graph.instantiate(parameters)

        self.input_feature_names = {}
        self.input_target_name = None

        final_component = self._component_graph.get_last_component()
        self.estimator = final_component if isinstance(final_component, Estimator) else None
        self._estimator_name = self._component_graph.compute_order[-1] if self.estimator is not None else None

        self._validate_estimator_problem_type()
        self._is_fitted = False
        self._pipeline_params = parameters.get("pipeline", {})

        self.data = pd.DataFrame({})
        for ct in range(1, 4):
            d = 'https://bciavm.s3.amazonaws.com/dfPricesEpc_parquet/data'+str(ct)+'.parquet'
            self.data = self.data.append(pd.read_parquet(d))


        t1, t2, y1, y2 = preprocess_data(self.data)
        t1[target] = y1
        t2[target] = y2
        self.data = pd.concat([t1,t2])
        self.target = target
        try:
            latest_version_info = client.get_latest_versions('avm', stages=[ "Production" ])
            self.latest_production_version = int(latest_version_info[ 0 ].version) + 1
        except:
            self.latest_production_version = np.nan
        try:
            latest_version_info = client.get_latest_versions('avm', stages=[ "Staging" ])
            self.latest_staging_version = int(latest_version_info[ 0 ].version) + 1
        except:
            self.latest_staging_version = np.nan

        self.DATA_DICT = DATA_DICT
        self.COLUMNS = [col for col in DATA_DICT]
        self.data = self.data.rename({'Postcode': 'POSTCODE'}, axis=1)
        self.data = self.data[self.COLUMNS]

    @classproperty
    def name(cls):
        """Returns a name describing the pipeline.
        By default, this will take the class name and add a space between each capitalized word (class name should be in Pascal Case). If the pipeline has a custom_name attribute, this will be returned instead.
        """
        if cls.custom_name:
            name = cls.custom_name
        else:
            rex = re.compile(r'(?<=[a-z])(?=[A-Z])')
            name = rex.sub(' ', cls.__name__)
            if name == cls.__name__:
                raise IllFormattedClassNameError("Pipeline Class {} needs to follow Pascal Case standards or `custom_name` must be defined.".format(cls.__name__))
        return name

    @classproperty
    def summary(cls):
        """Returns a short summary of the pipeline structure, describing the list of components used.
        Example: Logistic Regression Classifier w/ Simple Imputer + One Hot Encoder
        """
        component_graph = [handle_component_class(component_class) for component_class in copy.copy(cls.linearized_component_graph)]
        if len(component_graph) == 0:
            return "Empty Pipeline"
        summary = "Pipeline"
        component_graph[-1] = component_graph[-1]

        if inspect.isclass(component_graph[-1]) and issubclass(component_graph[-1], Estimator):
            estimator_class = component_graph.pop(-1)
            summary = estimator_class.name
        if len(component_graph) == 0:
            return summary
        component_names = [component_class.name for component_class in component_graph]
        return '{} w/ {}'.format(summary, ' + '.join(component_names))

    @classproperty
    def linearized_component_graph(cls):
        """Returns a component graph in list form. Note: this is not guaranteed to be in proper component computation order"""
        if isinstance(cls.component_graph, list):
            return cls.component_graph
        else:
            return [component_info[0] for component_info in cls.component_graph.values()]

    def _validate_estimator_problem_type(self):
        """Validates this pipeline's problem_type against that of the estimator from `self.component_graph`"""
        if self.estimator is None:  # Allow for pipelines that do not end with an estimator
            return
        estimator_problem_types = self.estimator.supported_problem_types
        if self.problem_type not in estimator_problem_types:
            raise ValueError("Problem type {} not valid for this component graph. Valid problem types include {}."
                             .format(self.problem_type, estimator_problem_types))

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise NotImplementedError('Slicing pipelines is currently not supported.')
        return self._component_graph[index]

    def __setitem__(self, index, value):
        raise NotImplementedError('Setting pipeline components is not supported.')

    def get_component(self, name):
        """Returns component by name

        Arguments:
            name (str): Name of component

        Returns:
            Component: Component to return

        """
        return self._component_graph.get_component(name)

    def describe(self, return_dict=False):
        """Outputs pipeline details including component parameters

        Arguments:
            return_dict (bool): If True, return dictionary of information about pipeline. Defaults to False.

        Returns:
            dict: Dictionary of all component parameters if return_dict is True, else None
        """
        log_title(logger, self.name)
        logger.info("Problem Type: {}".format(self.problem_type))
        logger.info("Model Family: {}".format(str(self.model_family)))

        if self._estimator_name in self.input_feature_names:
            logger.info("Number of features: {}".format(len(self.input_feature_names[self._estimator_name])))

        # Summary of steps
        log_subtitle(logger, "Pipeline Steps")

        pipeline_dict = {
            "name": self.name,
            "problem_type": self.problem_type,
            "model_family": self.model_family,
            "components": dict()
        }

        for number, component in enumerate(self._component_graph, 1):
            component_string = str(number) + ". " + component.name
            logger.info(component_string)
            pipeline_dict["components"].update({component.name: component.describe(print_name=False, return_dict=return_dict)})
        if return_dict:
            return pipeline_dict

    def compute_estimator_features(self, X, y=None):
        """Transforms the data by applying all pre-processing components.

        Arguments:
            X (ww.DataTable, pd.DataFrame): Input data to the pipeline to transform.

        Returns:
            ww.DataTable: New transformed features.
        """
        X_t = self._component_graph.compute_final_component_features(X, y=y)
        return X_t

    def avm(self, X, batch_sample_sets=1, latest_production_version=1, min_sample_sz=15, conf_min=.50) :
        """Main method for the BCI avm predictions. 

        Arguments:
            X (ww.DataTable, pd.DataFrame, or np.ndarray): Data of shape [n_samples, n_features]
            latest_production_version (int): The MLFlow latest production version - used if the programatic search fails
            min_sample_sz (int): The minimum random sample size of comparable properties to use in determining the confidence.
            batch_sample_sets (int): The number of random sample batches to use in determining the confidence.
        Returns:
            pd.DataFrame: Predicted values.
        """
        
        if 'unit_index' not in X.columns:
            X = X.reset_index().rename({'index':'unit_index'}, axis=1)

        self.min_sample_sz = min_sample_sz
        self.bss = batch_sample_sets
        if np.isnan(self.latest_production_version):
            self.latest_production_version = latest_production_version
            self.latest_staging_version = latest_production_version - 1

        X = infer_feature_types(X)
        self.l = [ ]
        self.model_input = X

        #computes the confidence + upper & lower bound predictions
        X[ 'unit_index' ].apply(self._conf)

        #builds and formats the final prediction response
        predictions = pd.concat(self.l, ignore_index=True)
        predictions[ 'avm' ] = round(predictions[ 'avm' ].astype(float), 0)
        try :
            predictions[ 'avm' ] = np.where(predictions[ 'avm' ].astype(float) < 0.0, np.nan, predictions[ 'avm' ].astype(float))
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
            predictions[ 'conf' ] = np.where(predictions[ 'conf' ].astype(float) < conf_min, np.nan,
                                         predictions[ 'conf' ].astype(float))
        except :
            pass
        return infer_feature_types(predictions)

    def _conf(self, unit_index):
        """Get the upper & lower bound predictions & confidence.

        Arguments:
            unit_index (int): The index of each property to predict.
        Returns:
            self
        """
        model_input = self.model_input
        unit = model_input[ model_input[ 'unit_index' ] == unit_index ]
        data = self.data
        cols = [col for col in unit.columns if col in data.columns]
        if self.target not in cols:
            cols.append(self.target)

        model_input = model_input.reset_index().drop('index', axis=1)
        unit = unit.reset_index().drop('index', axis=1)
        ts = str(datetime.utcnow().timestamp())
        latest_production_version = self.latest_production_version
        latest_staging_version = self.latest_staging_version
        y_predd, lower, upper, conf = [ np.nan ], np.nan, np.nan, np.nan
        lowers, uppers, confs, y_pred = [ ], [ ], [ ], [ ]
        match_key = unit[ 'POSTCODE_AREA' ].astype(str)
        match_key = match_key.values[ 0 ]
        dfs = data[ data[ 'POSTCODE_AREA' ]==match_key]
        unit = infer_feature_types(unit)
        y_predd = self._component_graph.predict(unit)

        if len(dfs) < self.min_sample_sz:
            resp = pd.DataFrame({})
            resp['unit_index'] = unit['unit_index']
            resp['avm'] = y_predd
            resp['avm_lower'] = [np.nan]
            resp['avm_upper'] = [np.nan]
            resp['conf'] = [np.nan]
            resp['ts'] = [ts]
            resp['latest_production_version'] = [latest_production_version]
            resp['latest_staging_version'] = [latest_staging_version]
            return resp

        comp_df = dfs.copy()
        y_trus_all, y_preds_all = [], []
        for n in range(self.bss):
            dfs = comp_df.sample(frac=.2, replace=True)
            if len(dfs) > 100:
                dfs = dfs.sample(100)

            y_trus = dfs[self.target]
            dfs = infer_feature_types(dfs)
            dfs = dfs.reset_index().drop('index', axis=1)
            y_preds = self._component_graph.predict(dfs)
            y_trus, y_preds = np.array(y_trus).astype(float), np.array(y_preds).astype(float)
            y_trus_all.extend(y_trus)
            y_preds_all.extend(y_preds)

        stats = (np.array([y for y in y_preds_all]) - y_trus_all) / y_trus_all
        stats = np.nan_to_num(stats)
        confs = [.90]
        for alpha in confs:
            p = ((1.0 - alpha) / 2.0) * 100.0
            ylower = np.percentile(stats, p)
            p = (alpha + ((1.0 - alpha) / 2.0)) * 100.0
            yupper = np.percentile(stats, p)
            if yupper > 0.0 and ylower < 0.0:
                upper = yupper
                lower = ylower
                conf = alpha
                break

        resp = pd.DataFrame({})
        resp[ 'unit_index' ] = unit[ 'unit_index' ]
        resp[ 'avm' ] = y_predd
        resp[ 'avm_lower' ] = [ lower ]
        resp[ 'avm_upper' ] = [ upper ]

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
        resp['ts'] = [ts]
        resp['latest_production_version'] = [latest_production_version]
        resp['latest_staging_version'] = [latest_staging_version]
        self.l.append(resp)
        return self


    def _compute_features_during_fit(self, X, y):
        self.input_target_name = y.name
        X_t = self._component_graph.fit_features(X, y)
        self.input_feature_names = self._component_graph.input_feature_names
        return X_t

    def _fit(self, X, y):
        self.input_target_name = y.name
        self._component_graph.fit(X, y)
        self.input_feature_names = self._component_graph.input_feature_names
        self._parallel_post_fit(X, y)

    def _parallel_post_fit(self, X, y=None):
        """Meta-estimator for parallel predict and transform.

        Warning
        This class is not appropriate for parallel or distributed training on large datasets.
        For that, see Incremental, which provides distributed (but sequential) training.
        If you’re doing distributed hyperparameter optimization on larger-than-memory datasets,
        see dask_ml.model_selection.IncrementalSearch.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, np.ndarray): The target training data of length [n_samples]

        Returns:
            Estimator wrapped in Dask ParallelPostFit
        """
        wrapped_model = AVMWrapper(self)
        wmodel = ParallelPostFit(wrapped_model)
        wmodel.fit(X, y)
        self._dask_component_graph = wmodel
        return self

    @abstractmethod
    def fit(self, X, y):
        """Build a model

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, np.ndarray): The target training data of length [n_samples]

        Returns:
            self

        """

    def predict(self, X, objective=None):
        """Make predictions using selected features.

        Arguments:
            X (ww.DataTable, pd.DataFrame, or np.ndarray): Data of shape [n_samples, n_features]
            objective (Object or string): The objective to use to make predictions

        Returns:
            ww.DataColumn: Predicted values.
        """
        X = fill_nulls(X)
        X = infer_feature_types(X)
        predictions = self._component_graph.predict(X)
        predictions.name = self.input_target_name
        return infer_feature_types(predictions)

    def parallel_predict(self, X, columns=['unit_index', 'avm', 'avm_lower', 'avm_upper', 'conf', 'ts',
                                                 'latest_production_version', 'latest_staging_version']):
        """Make predictions using selected features.

        Arguments:
            X (dd.DataFrame, pd.DataFrame, or np.ndarray): Data of shape [n_samples, n_features]
            columns (list): The list of columns to use when returning predictions
            scheduler: The dask scheduler type.
            n_jobs: The number of parallel jobs - should equal npartitions in the dask dataframe

        Returns:
            pd.DataFrame: Predicted values.
        """
        
        def predict(X, model):
            return model.predict(X)

        predictions = X.map_partitions(predict, model=self._dask_component_graph)
        return pd.DataFrame(predictions, columns=columns)

    @abstractmethod
    def score(self, X, y, objectives):
        """Evaluate model performance on current and additional objectives

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features]
            y (pd.Series, ww.DataColumn, or np.ndarray): True labels of length [n_samples]
            objectives (list): Non-empty list of objectives to score on

        Returns:
            dict: Ordered dictionary of objective scores
        """

    @staticmethod
    def _score(X, y, predictions, objective):
        return objective.score(y, predictions, X)

    def _score_all_objectives(self, X, y, y_pred, y_pred_proba, objectives):
        """Given data, model predictions or predicted probabilities computed on the data, and an objective, evaluate and return the objective score.

        Will raise a PipelineScoreError if any objectives fail.
        Arguments:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target data.
            y_pred (pd.Series): The pipeline predictions.
            y_pred_proba (pd.Dataframe, pd.Series, None): The predicted probabilities for classification problems.
                Will be a DataFrame for multiclass problems and Series otherwise. Will be None for regression problems.
            objectives (list): List of objectives to score.

        Returns:
            dict: Ordered dictionary with objectives and their scores.
        """
        scored_successfully = OrderedDict()
        exceptions = OrderedDict()
        for objective in objectives:
            try:
                if not objective.is_defined_for_problem_type(self.problem_type):
                    raise ValueError(f'Invalid objective {objective.name} specified for problem type {self.problem_type}')
                y_pred = self._select_y_pred_for_score(X, y, y_pred, y_pred_proba, objective)
                score = self._score(X, y, y_pred_proba if objective.score_needs_proba else y_pred, objective)
                scored_successfully.update({objective.name: score})
            except Exception as e:
                tb = traceback.format_tb(sys.exc_info()[2])
                exceptions[objective.name] = (e, tb)
        if exceptions:
            # If any objective failed, throw an PipelineScoreError
            raise PipelineScoreError(exceptions, scored_successfully)
        # No objectives failed, return the scores
        return scored_successfully

    def _select_y_pred_for_score(self, X, y, y_pred, y_pred_proba, objective):
        return y_pred

    @classproperty
    def model_family(cls):
        """Returns model family of this pipeline template"""
        component_graph = copy.copy(cls.component_graph)
        if isinstance(component_graph, list):
            return handle_component_class(component_graph[-1]).model_family
        else:
            order = ComponentGraph.generate_order(component_graph)
            final_component = order[-1]
            return handle_component_class(component_graph[final_component][0]).model_family

    @classproperty
    def hyperparameters(cls):
        """Returns hyperparameter ranges from all components as a dictionary"""
        hyperparameter_ranges = dict()
        component_graph = copy.copy(cls.component_graph)
        if isinstance(component_graph, list):
            for component_class in component_graph:
                component_class = handle_component_class(component_class)
                component_hyperparameters = copy.copy(component_class.hyperparameter_ranges)
                if cls.custom_hyperparameters and component_class.name in cls.custom_hyperparameters:
                    component_hyperparameters.update(cls.custom_hyperparameters.get(component_class.name, {}))
                hyperparameter_ranges[component_class.name] = component_hyperparameters
        else:
            for component_name, component_info in component_graph.items():
                component_class = handle_component_class(component_info[0])
                component_hyperparameters = copy.copy(component_class.hyperparameter_ranges)
                if cls.custom_hyperparameters and component_name in cls.custom_hyperparameters:
                    component_hyperparameters.update(cls.custom_hyperparameters.get(component_name, {}))
                hyperparameter_ranges[component_name] = component_hyperparameters
        return hyperparameter_ranges

    @property
    def parameters(self):
        """Returns parameter dictionary for this pipeline

        Returns:
            dict: Dictionary of all component parameters
        """
        components = [(component_name, component_class) for component_name, component_class in self._component_graph.component_instances.items()]
        component_parameters = {c_name: copy.copy(c.parameters) for c_name, c in components if c.parameters}
        if self._pipeline_params:
            component_parameters['pipeline'] = self._pipeline_params
        return component_parameters

    @classproperty
    def default_parameters(cls):
        """Returns the default parameter dictionary for this pipeline.

        Returns:
            dict: Dictionary of all component default parameters.
        """
        defaults = {}
        for c in cls.component_graph:
            component = handle_component_class(c)
            if component.default_parameters:
                defaults[component.name] = component.default_parameters
        return defaults

    @property
    def feature_importance(self):
        """Return importance associated with each feature. Features dropped by the feature selection are excluded.

        Returns:
            pd.DataFrame including feature names and their corresponding importance
        """
        feature_names = self.input_feature_names[self._estimator_name]
        importance = list(zip(feature_names, self.estimator.feature_importance))  # note: this only works for binary
        importance.sort(key=lambda x: -abs(x[1]))
        df = pd.DataFrame(importance, columns=["feature", "importance"])
        return df

    def graph(self, filepath=None):
        """Generate an image representing the pipeline graph

        Arguments:
            filepath (str, optional): Path to where the graph should be saved. If set to None (as by default), the graph will not be saved.

        Returns:
            graphviz.Digraph: Graph object that can be directly displayed in Jupyter notebooks.
        """
        graphviz = import_or_raise('graphviz', error_msg='Please install graphviz to visualize pipelines.')

        # Try rendering a dummy graph to see if a working backend is installed
        try:
            graphviz.Digraph().pipe()
        except graphviz.backend.ExecutableNotFound:
            raise RuntimeError(
                "To graph entity sets, a graphviz backend is required.\n" +
                "Install the backend using one of the following commands:\n" +
                "  Mac OS: brew install graphviz\n" +
                "  Linux (Ubuntu): sudo apt-get install graphviz\n" +
                "  Windows: conda install python-graphviz\n"
            )

        graph_format = None
        path_and_name = None
        if filepath:
            # Explicitly cast to str in case a Path object was passed in
            filepath = str(filepath)
            try:
                f = open(filepath, 'w')
                f.close()
            except (IOError, FileNotFoundError):
                raise ValueError(('Specified filepath is not writeable: {}'.format(filepath)))
            path_and_name, graph_format = os.path.splitext(filepath)
            graph_format = graph_format[1:].lower()  # ignore the dot
            supported_filetypes = graphviz.backend.FORMATS
            if graph_format not in supported_filetypes:
                raise ValueError(("Unknown format '{}'. Make sure your format is one of the " +
                                  "following: {}").format(graph_format, supported_filetypes))

        graph = self._component_graph.graph(path_and_name, graph_format)

        if filepath:
            graph.render(path_and_name, cleanup=True)

        return graph

    def graph_feature_importance(self, importance_threshold=0):
        """Generate a bar graph of the pipeline's feature importance

        Arguments:
            importance_threshold (float, optional): If provided, graph features with a permutation importance whose absolute value is larger than importance_threshold. Defaults to zero.

        Returns:
            plotly.Figure, a bar graph showing features and their corresponding importance
        """
        go = import_or_raise("plotly.graph_objects", error_msg="Cannot find dependency plotly.graph_objects")
        if jupyter_check():
            import_or_raise("ipywidgets", warning=True)

        feat_imp = self.feature_importance
        feat_imp['importance'] = abs(feat_imp['importance'])

        if importance_threshold < 0:
            raise ValueError(f'Provided importance threshold of {importance_threshold} must be greater than or equal to 0')

        # Remove features with importance whose absolute value is less than importance threshold
        feat_imp = feat_imp[feat_imp['importance'] >= importance_threshold]

        # List is reversed to go from ascending order to descending order
        feat_imp = feat_imp.iloc[::-1]

        title = 'Feature Importance'
        subtitle = 'May display fewer features due to feature selection'
        data = [go.Bar(
            x=feat_imp['importance'],
            y=feat_imp['feature'],
            orientation='h'
        )]

        layout = {
            'title': '{0}<br><sub>{1}</sub>'.format(title, subtitle),
            'height': 800,
            'xaxis_title': 'Feature Importance',
            'yaxis_title': 'Feature',
            'yaxis': {
                'type': 'category'
            }
        }

        fig = go.Figure(data=data, layout=layout)
        return fig

    def save(self, file_path, pickle_protocol=cloudpickle.DEFAULT_PROTOCOL):
        """Saves pipeline at file path

        Arguments:
            file_path (str): location to save file
            pickle_protocol (int): the pickle data stream format.

        Returns:
            None
        """
        with open(file_path, 'wb') as f:
            cloudpickle.dump(self, f, protocol=pickle_protocol)

    @staticmethod
    def load(file_path):
        """Loads pipeline at file path

        Arguments:
            file_path (str): location to load file

        Returns:
            PipelineBase object
        """
        with open(file_path, 'rb') as f:
            return cloudpickle.load(f)

    def clone(self):
        """Constructs a new pipeline with the same components, parameters, and random state.

        Returns:
            A new instance of this pipeline with identical components, parameters, and random state.
        """
        return self.__class__(self.parameters, random_seed=self.random_seed)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        random_seed_eq = self.random_seed == other.random_seed
        if not random_seed_eq:
            return False
        attributes_to_check = ['parameters', '_is_fitted', 'component_graph', 'input_feature_names', 'input_target_name']
        for attribute in attributes_to_check:
            if getattr(self, attribute) != getattr(other, attribute):
                return False
        return True

    def __str__(self):
        return self.name

    def __repr__(self):

        def repr_component(parameters):
            return ', '.join([f"'{key}': {safe_repr(value)}" for key, value in parameters.items()])

        parameters_repr = ' '.join([f"'{component}':{{{repr_component(parameters)}}}," for component, parameters in self.parameters.items()])
        return f'{(type(self).__name__)}(parameters={{{parameters_repr}}})'

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._component_graph)

    def _get_feature_provenance(self):
        return self._component_graph._feature_provenance

    @property
    def _supports_fast_permutation_importance(self):
        has_more_than_one_estimator = sum(isinstance(c, Estimator) for c in self._component_graph) > 1
        _all_components = set(all_components())
        has_custom_components = any(c.__class__ not in _all_components for c in self._component_graph)
        has_dim_reduction = any(isinstance(c, (PCA, LinearDiscriminantAnalysis)) for c in self._component_graph)
        has_dfs = any(isinstance(c, DFSTransformer) for c in self._component_graph)
        has_stacked_ensembler = any(isinstance(c, ( StackedEnsembleRegressor)) for c in self._component_graph)
        return not any([has_more_than_one_estimator, has_custom_components, has_dim_reduction, has_dfs, has_stacked_ensembler])

    @staticmethod
    def create_objectives(objectives):
        objective_instances = []
        for objective in objectives:
            try:
                objective_instances.append(get_objective(objective, return_instance=True))
            except ObjectiveCreationError as e:
                msg = f"Cannot pass {objective} as a string in pipeline.score. Instantiate first and then add it to the list of objectives."
                raise ObjectiveCreationError(msg) from e
        return objective_instances

    def can_tune_threshold_with_objective(self, objective):
        """Determine whether the threshold of a binary classification pipeline can be tuned.

       Arguments:
            pipeline (PipelineBase): Binary classification pipeline.
            objective (ObjectiveBase): Primary AutoMLSearch objective.

        Returns:
            bool: True if the pipeline threshold can be tuned.

        """
        return is_binary(self.problem_type) and objective.is_defined_for_problem_type(self.problem_type) and objective.can_optimize_threshold
