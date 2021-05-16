import importlib
import os
import warnings
from collections import namedtuple
from functools import reduce

import numpy as np
import pandas as pd
import woodwork as ww
from sklearn.utils import check_random_state

from bciavm.exceptions import (
    EnsembleMissingPipelinesError,
    MissingComponentError
)
from bciavm.utils import get_logger

logger = get_logger(__file__)


def import_or_raise(library, error_msg=None, warning=False):
    """Attempts to import the requested library by name.
    If the import fails, raises an ImportError or warning.

    Arguments:
        library (str): the name of the library
        error_msg (str): error message to return if the import fails
        warning (bool): if True, import_or_raise gives a warning instead of ImportError. Defaults to False.
    """
    try:
        return importlib.import_module(library)
    except ImportError:
        if error_msg is None:
            error_msg = ""
        msg = (f"Missing optional dependency '{library}'. Please use pip to install {library}. {error_msg}")
        if warning:
            warnings.warn(msg)
        else:
            raise ImportError(msg)
    except Exception as ex:
        msg = (f"An exception occurred while trying to import `{library}`: {str(ex)}")
        if warning:
            warnings.warn(msg)
        else:
            raise Exception(msg)


def convert_to_seconds(input_str):
    """Converts a string describing a length of time to its length in seconds."""
    hours = {'h', 'hr', 'hour', 'hours'}
    minutes = {'m', 'min', 'minute', 'minutes'}
    seconds = {'s', 'sec', 'second', 'seconds'}
    value, unit = input_str.split()
    if unit[-1] == 's' and len(unit) != 1:
        unit = unit[:-1]
    if unit in seconds:
        return float(value)
    elif unit in minutes:
        return float(value) * 60
    elif unit in hours:
        return float(value) * 3600
    else:
        msg = "Invalid unit. Units must be hours, mins, or seconds. Received '{}'".format(unit)
        raise AssertionError(msg)


# specifies the min and max values a seed to np.random.RandomState is allowed to take.
# these limits were chosen to fit in the numpy.int32 datatype to avoid issues with 32-bit systems
# see https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.RandomState.html
SEED_BOUNDS = namedtuple('SEED_BOUNDS', ('min_bound', 'max_bound'))(0, 2**31 - 1)


def get_random_state(seed):
    """Generates a numpy.random.RandomState instance using seed.

    Arguments:
        seed (None, int, np.random.RandomState object): seed to use to generate numpy.random.RandomState. Must be between SEED_BOUNDS.min_bound and SEED_BOUNDS.max_bound, inclusive. Otherwise, an exception will be thrown.
    """
    if isinstance(seed, (int, np.integer)) and (seed < SEED_BOUNDS.min_bound or SEED_BOUNDS.max_bound < seed):
        raise ValueError('Seed "{}" is not in the range [{}, {}], inclusive'.format(seed, SEED_BOUNDS.min_bound, SEED_BOUNDS.max_bound))
    return check_random_state(seed)


def get_random_seed(random_state, min_bound=SEED_BOUNDS.min_bound, max_bound=SEED_BOUNDS.max_bound):
    """Given a numpy.random.RandomState object, generate an int representing a seed value for another random number generator. Or, if given an int, return that int.

    To protect against invalid input to a particular library's random number generator, if an int value is provided, and it is outside the bounds "[min_bound, max_bound)", the value will be projected into the range between the min_bound (inclusive) and max_bound (exclusive) using modular arithmetic.

    Arguments:
        random_state (int, numpy.random.RandomState): random state
        min_bound (None, int): if not default of None, will be min bound when generating seed (inclusive). Must be less than max_bound.
        max_bound (None, int): if not default of None, will be max bound when generating seed (exclusive). Must be greater than min_bound.

    Returns:
        int: seed for random number generator
    """
    if not min_bound < max_bound:
        raise ValueError("Provided min_bound {} is not less than max_bound {}".format(min_bound, max_bound))
    if isinstance(random_state, np.random.RandomState):
        return random_state.randint(min_bound, max_bound)
    if random_state < min_bound or random_state >= max_bound:
        return ((random_state - min_bound) % (max_bound - min_bound)) + min_bound
    return random_state


class classproperty:
    """Allows function to be accessed as a class level property.
        Example:
        class LogisticRegressionBinaryPipeline(PipelineBase):
            component_graph = ['Simple Imputer', 'Logistic Regression Classifier']

            @classproperty
            def summary(cls):
            summary = ""
            for component in cls.component_graph:
                component = handle_component_class(component)
                summary += component.name + " + "
            return summary

            assert LogisticRegressionBinaryPipeline.summary == "Simple Imputer + Logistic Regression Classifier + "
            assert LogisticRegressionBinaryPipeline().summary == "Simple Imputer + Logistic Regression Classifier + "
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, _, klass):
        return self.func(klass)


def _get_subclasses(base_class):
    """Gets all of the leaf nodes in the hiearchy tree for a given base class.

    Arguments:
        base_class (abc.ABCMeta): Class to find all of the children for.

    Returns:
        subclasses (list): List of all children that are not base classes.
    """

    classes_to_check = base_class.__subclasses__()
    subclasses = []

    while classes_to_check:
        subclass = classes_to_check.pop()
        children = subclass.__subclasses__()

        if children:
            classes_to_check.extend(children)
        else:
            subclasses.append(subclass)

    return subclasses


_not_used_in_automl = {'BaselineClassifier', 'BaselineRegressor', 'TimeSeriesBaselineEstimator',
                       'StackedEnsembleClassifier', 'StackedEnsembleRegressor',
                       'ModeBaselineBinaryPipeline', 'BaselineBinaryPipeline', 'MeanBaselineRegressionPipeline',
                       'BaselineRegressionPipeline', 'ModeBaselineMulticlassPipeline', 'BaselineMulticlassPipeline',
                       'TimeSeriesBaselineRegressionPipeline', 'TimeSeriesBaselineBinaryPipeline',
                       'TimeSeriesBaselineMulticlassPipeline', 'KNeighborsClassifier',
                       'SVMClassifier', 'SVMRegressor', 'ARIMARegressor'}


def get_importable_subclasses(base_class, used_in_automl=True):
    """Get importable subclasses of a base class. Used to list all of our
    estimators, transformers, components and pipelines dynamically.

    Arguments:
        base_class (abc.ABCMeta): Base class to find all of the subclasses for.
        args (list): Args used to instantiate the subclass. [{}] for a pipeline, and [] for
            all other classes.
        used_in_automl: Not all components/pipelines/estimators are used in automl search. If True,
            only include those subclasses that are used in the search. This would mean excluding classes related to
            ExtraTrees, ElasticNet, and Baseline estimators.

    Returns:
        List of subclasses.
    """
    all_classes = _get_subclasses(base_class)

    classes = []
    for cls in all_classes:
        if 'bci_avm.pipelines' not in cls.__module__:
            continue
        try:
            cls()
            classes.append(cls)
        except (ImportError, MissingComponentError, TypeError):
            logger.debug(f'Could not import class {cls.__name__} in get_importable_subclasses')
        except EnsembleMissingPipelinesError:
            classes.append(cls)
    if used_in_automl:
        classes = [cls for cls in classes if cls.__name__ not in _not_used_in_automl]

    return classes


def _rename_column_names_to_numeric(X, flatten_tuples=True):
    """Used in LightGBM and XGBoost estimator classes to rename column names
        when the input is a pd.DataFrame in case it has column names that contain symbols ([, ], <)
        that these estimators cannot natively handle.

    Arguments:
        X (pd.DataFrame): The input training data of shape [n_samples, n_features]
        flatten_tuples (bool): Whether to flatten MultiIndex or tuple column names. LightGBM cannot handle columns with tuple names.

    Returns:
        Transformed X where column names are renamed to numerical values
    """
    if isinstance(X, (np.ndarray, list)):
        return pd.DataFrame(X)

    if flatten_tuples and (len(X.columns) > 0 and isinstance(X.columns, pd.MultiIndex)):
        flat_col_names = list(map(str, X.columns))
        X.columns = flat_col_names
        rename_cols_dict = dict((str(col), col_num) for col_num, col in enumerate(list(X.columns)))
    else:
        rename_cols_dict = dict((col, col_num) for col_num, col in enumerate(list(X.columns)))
    X_renamed = X.rename(columns=rename_cols_dict)
    return X_renamed


def jupyter_check():
    """Get whether or not the code is being run in a Ipython environment (such as Jupyter Notebook or Jupyter Lab)

    Arguments:
        None

    Returns:
        Boolean: True if Ipython, False otherwise
    """
    try:
        ipy = import_or_raise("IPython")
        return ipy.core.getipython.get_ipython()
    except Exception:
        return False


def safe_repr(value):
    """Convert the given value into a string that can safely be used for repr

    Arguments:
        value: the item to convert

    Returns:
        String representation of the value
    """
    if isinstance(value, float):
        if pd.isna(value):
            return 'np.nan'
        if np.isinf(value):
            return f"float('{repr(value)}')"
    return repr(value)


def is_all_numeric(dt):
    """Checks if the given DataTable contains only numeric values

    Arguments:
        dt (pd.DataFrame): The DataTable to check data types of.

    Returns:
        True if all the columns are numeric and are not missing any values, False otherwise.
    """
    for col_tags in dt.ww.semantic_tags.values():
        if "numeric" not in col_tags:
            return False

    if dt.isnull().any().any():
        return False
    return True


def pad_with_nans(pd_data, num_to_pad):
    """Pad the beginning num_to_pad rows with nans.

    Arguments:
        pd_data (pd.DataFrame or pd.Series): Data to pad.

    Returns:
        pd.DataFrame or pd.Series
    """
    if isinstance(pd_data, pd.Series):
        padding = pd.Series([np.nan] * num_to_pad, name=pd_data.name)
    else:
        padding = pd.DataFrame({col: [np.nan] * num_to_pad
                                for col in pd_data.columns})
    padded = pd.concat([padding, pd_data], ignore_index=True)
    # By default, pd.concat will convert all types to object if there are mixed numerics and objects
    # The call to convert_dtypes ensures numerics stay numerics in the new dataframe.
    return padded.convert_dtypes(infer_objects=True, convert_string=False,
                                 convert_integer=False, convert_boolean=False)


def _get_rows_without_nans(*data):
    """Compute a boolean array marking where all entries in the data are non-nan.

    Arguments:
        *data (sequence of pd.Series or pd.DataFrame)

    Returns:
        np.ndarray: mask where each entry is True if and only if all corresponding entries in that index in data
            are non-nan.
    """
    def _not_nan(pd_data):
        if pd_data is None or len(pd_data) == 0:
            return np.array([True])
        if isinstance(pd_data, pd.Series):
            return ~pd_data.isna().values
        elif isinstance(pd_data, pd.DataFrame):
            return ~pd_data.isna().any(axis=1).values
        else:
            return pd_data

    mask = reduce(lambda a, b: np.logical_and(_not_nan(a), _not_nan(b)), data)
    return mask


def drop_rows_with_nans(*pd_data):
    """Drop rows that have any NaNs in all dataframes or series.

    Arguments:
        *pd_data (sequence of pd.Series or pd.DataFrame or None)

    Returns:
        list of pd.DataFrame or pd.Series or None
    """

    mask = _get_rows_without_nans(*pd_data)

    def _subset(pd_data):
        if pd_data is not None and not pd_data.empty:
            return pd_data.iloc[mask]
        return pd_data

    return [_subset(data) for data in pd_data]


def _file_path_check(filepath=None, format='png', interactive=False, is_plotly=False):
    """ Helper function to check the filepath being passed.

    Arguments:
        filepath (str or Path, optional): Location to save file.
        format (str): Extension for figure to be saved as. Defaults to 'png'.
        interactive (bool, optional): If True and fig is of type plotly.Figure, sets the format to 'html'.
        is_plotly (bool, optional): Check to see if the fig being passed is of type plotly.Figure.

    Returns:
        String representing the final filepath the image will be saved to.
    """
    if filepath:
        filepath = str(filepath)
        path_and_name, extension = os.path.splitext(filepath)
        extension = extension[1:].lower() if extension else None
        if is_plotly and interactive:
            format_ = 'html'
        elif not extension and not interactive:
            format_ = format
        else:
            format_ = extension
        filepath = f'{path_and_name}.{format_}'
        try:
            f = open(filepath, 'w')
            f.close()
        except (IOError, FileNotFoundError):
            raise ValueError(('Specified filepath is not writeable: {}'.format(filepath)))
    return filepath


def save_plot(fig, filepath=None, format='png', interactive=False, return_filepath=False):
    """Saves fig to filepath if specified, or to a default location if not.

    Arguments:
        fig (Figure): Figure to be saved.
        filepath (str or Path, optional): Location to save file. Default is with filename "test_plot".
        format (str): Extension for figure to be saved as. Ignored if interactive is True and fig
        is of type plotly.Figure. Defaults to 'png'.
        interactive (bool, optional): If True and fig is of type plotly.Figure, saves the fig as interactive
        instead of static, and format will be set to 'html'. Defaults to False.
        return_filepath (bool, optional): Whether to return the final filepath the image is saved to. Defaults to False.

    Returns:
        String representing the final filepath the image was saved to if return_filepath is set to True.
        Defaults to None.
    """
    plotly_ = import_or_raise("plotly", error_msg="Cannot find dependency plotly")
    graphviz_ = import_or_raise('graphviz', error_msg='Please install graphviz to visualize trees.')
    matplotlib = import_or_raise("matplotlib", error_msg="Cannot find dependency matplotlib")
    plt_ = matplotlib.pyplot
    axes_ = matplotlib.axes

    is_plotly = False
    is_graphviz = False
    is_plt = False
    is_seaborn = False

    format = format if format else 'png'
    if isinstance(fig, plotly_.graph_objects.Figure):
        is_plotly = True
    elif isinstance(fig, graphviz_.Source):
        is_graphviz = True
    elif isinstance(fig, plt_.Figure):
        is_plt = True
    elif isinstance(fig, axes_.SubplotBase):
        is_seaborn = True

    if not filepath:
        extension = 'html' if interactive and is_plotly else format
        filepath = os.path.join(os.getcwd(), f'test_plot.{extension}')

    filepath = _file_path_check(filepath, format=format, interactive=interactive, is_plotly=is_plotly)

    if is_plotly and interactive:
        fig.write_html(file=filepath)
    elif is_plotly and not interactive:
        fig.write_image(file=filepath, engine="kaleido")
    elif is_graphviz:
        filepath_, format_ = os.path.splitext(filepath)
        fig.format = 'png'
        filepath = f'{filepath_}.png'
        fig.render(filename=filepath_, view=False, cleanup=True)
    elif is_plt:
        fig.savefig(fname=filepath)
    elif is_seaborn:
        fig = fig.figure
        fig.savefig(fname=filepath)

    if return_filepath:
        return filepath


def deprecate_arg(old_arg, new_arg, old_value, new_value):
    """Helper to raise warnings when a deprecated arg is used.

    Arguments:
        old_arg (str): Name of old/deprecated argument.
        new_arg (str): Name of new argument.
        old_value (Any): Value the user passed in for the old argument.
        new_value (Any): Value the user passed in for the new argument.

    Returns:
        old_value if not None, else new_value
    """
    value_to_use = new_value
    if old_value is not None:
        warnings.warn(f"Argument '{old_arg}' has been deprecated in favor of '{new_arg}'. "
                      f"Passing '{old_arg}' in future versions will result in an error.")
        value_to_use = old_value
    return value_to_use


def fill_nulls(X, DATA_DICT = {
                       "unit_id": {
                          "Name": "unit_id",
                          "Type": "long",
                          "PandasType": "int64",
                          "LogicalType": "Integer",
                          "SemanticType": "drop",
                          "Descriptions": "Unit ID",
                          "Item": "nan",
                          "Example": -1
                       },
                       "project_id": {
                          "Name": "project_id",
                          "Type": "long",
                          "PandasType": "float64",
                           "LogicalType": "Double",
                           "SemanticType": "drop",
                          "Descriptions": "Project ID",
                          "Item": "nan",
                          "Example": -1.0
                       },
                       "city_id": {
                          "Name": "city_id",
                          "Type": "long",
                          "PandasType": "float64",
                           "LogicalType": "Double",
                           "SemanticType": "drop",
                          "Descriptions": "Location ID",
                          "Item": "nan",
                          "Example": -1.0
                       },
                       "region_id": {
                          "Name": "region_id",
                          "Type": "long",
                          "PandasType": "float64",
                           "LogicalType": "Double",
                           "SemanticType": "drop",
                          "Descriptions": "Region ID",
                          "Item": "nan",
                          "Example": -1.0
                       },
                       "country_id": {
                          "Name": "country_id",
                          "Type": "long",
                          "PandasType": "float64",
                           "LogicalType": "Double",
                           "SemanticType": "drop",
                          "Descriptions": "Country ID",
                          "Item": "nan",
                          "Example": 212.0
                       },
                       "lat_lng": {
                          "Name": "lat_lng",
                          "Type": "geo_point",
                          "PandasType": "object",
                          "LogicalType": "NaturalLanguage",
                           "SemanticType": "drop",
                          "Descriptions": "nan",
                          "Item": "nan",
                          "Example": "{'lat':-1.0,'lng':-1.0}"
                       },
                       "description": {
                          "Name": "description",
                          "Type": "text",
                          "PandasType": "object",
                          "LogicalType": "NaturalLanguage",
                           "SemanticType": "drop",
                          "Descriptions": "Description",
                          "Item": "nan",
                          "Example": "<p>Seven Seas Cote (Phase 1) is a condo development located in Jomtien, Pattaya, Thailand. It is 1.1km from the beach with 1308 condo units across 8 floors scheduled for completion in December of 2018. It is hotel-managed and features an on-site restaurant, a swimming pool, a fitness center, a clubhouse, 24-hour security, CCTV, a reception/lobby area, a shuttle to the beach, car parking, a spa, a sauna.The Ultimate Luxury life style of the Mediterranean coast of South France. Bringing the most exclusive of Cannes, Monaco, Saint Tropez, Monte Carlo, Nice, and Marseilles in your city. Glamorous yachts, Marinas along the coast, beach clubs, golf playground, private beaches, Palace des festivals (film festival, jazz, shows), exclusive boulevard style shopping promenades with French style Restaurants, cafes, &amp; bistros.<\/p>"
                       },
                       "features": {
                          "Name": "features",
                          "Type": "long",
                          "PandasType": "object",
                          "LogicalType": "NaturalLanguage",
                           "SemanticType": "drop",
                          "Descriptions": "Array of features",
                          "Item": {'-1':'None',
                                   '1':'Beachfront',
                                   '2':'Beach Access',
                                   '3':'Oceanfront',
                                   '4':'Ocean Access',
                                   '5':'Customisable Layout',
                                   '6':'Media Room/Cinema',
                                   '7':'Private Gym',
                                   '8':'Private Lift',
                                   '9':'Pool Bar',
                                   '10':'Wet Bar',
                                   '11':'Private Sauna',
                                   '12':'Two Pools',
                                   '13':'Private Pool',
                                   '14':'Pool Access',
                                   '15':'Jacuzzi',
                                   '16':'Rooftop Terrace',
                                   '17':'Rooftop Garden',
                                   '18':'Private Garden',
                                   '19':'Garden Access',
                                   '20':'Pool Lounge',
                                   '21':'Terrace',
                                   '22':'Pond',
                                   '23':'Covered Parking',
                                   '24':'Corner Unit',
                                   '25':'Golf Membership',
                                   '26':'Maids Quarters',
                                   '27':'Duplex',
                                   '28':'Balcony',
                                   '29':'Wifi Included',
                                   '30':'Outdoor Sala',
                                   '31':'Outdoor Showers',
                                   '32':'Sunbeds Included',
                                   '33':'Safe Available',
                                   '34':'Full Western Kitchen',
                                   '35':'test',
                                   '36':'Bathtub',
                                   '37':'Fully Renovated',
                                   '38':'Renovated Kitchen',
                                   '39':'Renovated Bathroom',
                                   '40':'Washing Machine',
                                   '41':'Microwave',
                                   '42':'Oven',
                                   '43':'TV',
                                   '44':'Cable TV',
                                   '45':'Truevision',
                                   '46':'Gardening Included',
                                   '47':'Pool Cleaning Included',
                                   '48':'Direct access to BTS/MRT',
                                   '49':'Access to BTS/MRT'},
                          "Example": '-1'
                       },
                       "scenery": {
                          "Name": "scenery",
                          "Type": "long",
                          "PandasType": "object",
                          "LogicalType": "NaturalLanguage",
                           "SemanticType": "drop",
                          "Descriptions": "Array of scenery",
                          "Item": {'-1':'None',
                                   '1':'City View',
                                   '2':'Garden View',
                                   '3':'Golf Course View',
                                   '4':'Lake View',
                                   '5':'Mountain View',
                                   '6':'Pool View',
                                   '7':'Sea View',
                                   '8':'Blocked View',
                                   '9':'Park View',
                                   '10':'Partial Sea View',
                                   '11':'River or Canal View',
                                   '12':'Unblocked Open View'},
                          "Example": '-1'
                       },
                       "project_features": {
                          "Name": "project_features",
                          "Type": "long",
                          "PandasType": "object",
                          "LogicalType": "NaturalLanguage",
                           "SemanticType": "drop",
                          "Descriptions": "Array of project features",
                          "Item": {'-1':'None',
                                   '1':'Hotel Managed',
                                   '2':'On Site Restaurant',
                                   '3':'Communal Pool',
                                   '4':'Jacuzzi',
                                   '5':'Communal Gym',
                                   '6':'Bar',
                                   '7':'Clubhouse',
                                   '8':'24H Security',
                                   '9':'CCTV',
                                   '10':'Shuttle Bus To Beach',
                                   '11':'Kids Club',
                                   '12':'Car Parking',
                                   '13':'Spa',
                                   '14':'Steam Room',
                                   '15':'Direct Beach Access',
                                   '16':'Sauna',
                                   '17':'Tennis Court',
                                   '18':'Reception Lobby Area'},
                          "Example": '-1'
                       },
                       "furniture_option": {
                          "Name": "furniture_option",
                          "Type": "long",
                          "PandasType": "object",
                          "LogicalType": "Categorical",
                          "Descriptions": "Furnished option",
                          "Item": {'-1':'None',
                                   '1':'Unfurnished',
                                   '2':'Built In',
                                   '3':'Fully Furnished',
                                   '4':'Partly Furnished',
                                   '5':'Negotiable'},
                          "Example": '-1'
                       },
                       "owner_stay": {
                          "Name": "owner_stay",
                          "Type": "long",
                          "PandasType": "float64",
                           "LogicalType": "Double",
                          "Descriptions": "Allow owner to stay in a year",
                          "Item": "nan",
                          "Example": -1.0
                       },
                       "sinking_fund": {
                          "Name": "sinking_fund",
                          "Type": "long",
                          "PandasType": "float64",
                           "LogicalType": "Double",
                          "Descriptions": "Sinking fund",
                          "Item": "nan",
                          "Example": -1.0
                       },
                       "parking_spot": {
                          "Name": "parking_spot",
                          "Type": "long",
                          "PandasType": "int64",
                          "LogicalType": "Integer",
                          "Descriptions": "Number of private car parking, 0 = none",
                          "Item": "nan",
                          "Example": -1
                       },
                       "thumbnail": {
                          "Name": "thumbnail",
                          "Type": "text",
                          "PandasType": "object",
                          "LogicalType": "NaturalLanguage",
                           "SemanticType": "drop",
                          "Descriptions": "Thumnail url",
                          "Item": "nan",
                          "Example": "https://media.fazwaz.com/gallery/138521/c/1bedroom-rooms-big-3-1-banner-property.jpg"
                       },
                       "floor_number": {
                          "Name": "floor_number",
                          "Type": "long",
                          "PandasType": "float64",
                           "LogicalType": "Double",
                          "Descriptions": "For condo, what floor is this unit on",
                          "Item": "nan",
                          "Example": -1.0
                       },
                       "floors": {
                          "Name": "floors",
                          "Type": "long",
                          "PandasType": "float64",
                           "LogicalType": "Double",
                          "Descriptions": "For villa, how many floor does this unit have",
                          "Item": "nan",
                          "Example": -1.0
                       },
                       "views_count": {
                          "Name": "views_count",
                          "Type": "long",
                          "PandasType": "float64",
                           "LogicalType": "Double",
                          "Descriptions": "nan",
                          "Item": "nan",
                          "Example": -1.0
                       },
                       "property_type_id": {
                          "Name": "property_type_id",
                          "Type": "long",
                          "PandasType": "object",
                          "LogicalType": "Categorical",
                           "SemanticType": "drop",
                          "Descriptions": "Property type name",
                          "Item":  {'-1':'None',
                                    '6':'House',
                                    '7':'Villa',
                                    '8':'Condo',
                                    '9':'Apartment',
                                    '10':'Townhouse',
                                    '11':'Penthouse',
                                    '12':'Land',
                                    '13':'Private Island',
                                    '14':'Hotel',
                                    '15':'Restaurant',
                                    '16':'Bar',
                                    '17':'Shop',
                                    '18':'Office',
                                    '19':'Warehouse'},
                          "Example": '-1'
                       },
                       "property_type_name": {
                          "Name": "property_type_name",
                          "Type": "text",
                          "PandasType": "object",
                          "LogicalType": "Categorical",
                           "SemanticType": "drop",
                          "Descriptions": "Property type name",
                          "Item":  'nan',
                          "Example": "None"
                       },
                       "region_name": {
                          "Name": "region_name",
                          "Type": "text",
                          "PandasType": "object",
                          "LogicalType": "Categorical",
                           "SemanticType": "drop",
                          "Descriptions": "Region name",
                          "Item":  'nan',
                          "Example": "None"
                       },
                       "sinking_fund_option": {
                          "Name": "sinking_fund_option",
                          "Type": "long",
                          "PandasType": "object",
                          "LogicalType": "Categorical",
                          "Descriptions": "Sinking fund option",
                          "Item": {'-1':'None',
                                   '1' : 'Indoor Area',
                                   '2' : 'Outdoor Area',
                                   '3' : 'Indoor Area + Outdoor Area',
                                   '4' : 'Plot Size',
                                   '5' : 'Fixed Amount',
                                   '6' : 'Percentage'},
                          "Example": '-1'
                       },
                       "kitchen_option": {
                          "Name": "kitchen_option",
                          "Type": "long",
                          "PandasType": "object",
                          "LogicalType": "Categorical",
                          "Descriptions": "Kitchen furnished option",
                          "Item": {'-1':'None',
                                   '1':'Unfurnished',
                                   '2':'Built In',
                                   '3':'Fully Furnished',
                                   '4':'Partly Furnished',
                                   '5':'Negotiable'},
                          "Example": '-1'
                       },
                       "cam_fee_option": {
                          "Name": "cam_fee_option",
                          "Type": "long",
                          "PandasType": "object",
                          "LogicalType": "Categorical",
                          "Descriptions": "Cam fee option",
                          "Item": {'-1':'None',
                                   '1' : 'Indoor Area',
                                   '2' : 'Outdoor Area',
                                   '3' : 'Indoor Area + Outdoor Area',
                                   '4' : 'Plot Size',
                                   '5' : 'Fixed Amount',
                                   '6' : 'Percentage'},
                          "Example": '-1'
                       },
                       "is_allow_local_owner": {
                          "Name": "is_allow_local_owner",
                          "Type": "boolean",
                           "PandasType": "boolean",
                           "LogicalType": "Boolean",
                          "Descriptions": "Local owner",
                          "Item": "nan",
                          "Example": False
                       },
                       "is_allow_foreign_owner": {
                          "Name": "is_allow_foreign_owner",
                          "Type": "boolean",
                           "PandasType": "boolean",
                           "LogicalType": "Boolean",
                          "Descriptions": "Foreign owner",
                          "Item": "nan",
                          "Example": False
                       },
                       "is_self_guar_return": {
                          "Name": "is_self_guar_return",
                          "Type": "long",
                          "PandasType": "int64",
                          "LogicalType": "Integer",
                          "Descriptions": "Overwrite Guar. Rental",
                          "Item": {'-1':'None', '0' : 'no', '1' : 'yes'},
                          "Example": '-1'
                       },
                       "area": {
                          "Name": "area",
                          "Type": "float",
                           "PandasType": "float64",
                           "LogicalType": "Double",
                          "Descriptions": "Unit size in SQM",
                          "Item": "nan",
                          "Example": -1.0
                       },
                       "bedrooms": {
                          "Name": "bedrooms",
                          "Type": "long",
                          "PandasType": "float64",
                          "LogicalType": "Double",
                          "Descriptions": "Number of bedroom",
                          "Item": "nan",
                          "Example": -1.0
                       },
                       "bathrooms": {
                          "Name": "bathrooms",
                          "Type": "long",
                          "PandasType": "float64",
                          "LogicalType": "Double",
                          "Descriptions": "Number of bathroom",
                          "Item": "nan",
                          "Example": -1.0
                       },
                       "plot_size": {
                          "Name": "plot_size",
                          "Type": "float",
                           "PandasType": "float64",
                           "LogicalType": "Double",
                          "Descriptions": "Plot size in SQM",
                          "Item": "nan",
                          "Example": -1.0
                       },
                       "cam_fee": {
                          "Name": "cam_fee",
                          "Type": "long",
                          "PandasType": "float64",
                           "LogicalType": "Double",
                          "Descriptions": "CAM fee",
                          "Item": "nan",
                          "Example": 50
                       },
                       "created_at": {
                          "Name": "created_at",
                          "Type": "date",
                          "PandasType": "datetime64[ns]",
                          "LogicalType": "Datetime",
                           "SemanticType": "drop",
                          "Descriptions": "Created date",
                          "Item": "nan",
                          "Example": "2000-01-31T14:27:04+07:00"
                       },
                       "completion_date": {
                          "Name": "completion_date",
                          "Type": "date",
                          "PandasType": "datetime64[ns]",
                          "LogicalType": "Datetime",
                           "SemanticType": "drop",
                          "Descriptions": "Completion date of the project",
                          "Item": "nan",
                          "Example": "2000-01-01"
                       },
                       "sale_price": {
                          "Name": "sale_price",
                          "Type": "float",
                           "PandasType": "float64",
                           "LogicalType": "Double",
                          "Descriptions": "Sale price",
                          "Item": "nan",
                          "Example": 92367.26
                       }}):
    for col in X.columns:
        if col in DATA_DICT:
            X[col] = X[col].fillna(DATA_DICT[col]['Example'])

    return X


def calc_nearest_area_aggregate(X):
    X['nearest_area_class_median'] = np.where(( abs(X['area'] - X['area_median']) <= abs(X['area'] - X['area_max'])) | (abs(X['area'] - X['area_median']) <= abs(X['area'] - X['area_min'])), True, False)
    X['nearest_area_class_min'] = np.where(( abs(X['area'] - X['area_min']) <= abs(X['area'] - X['area_max'])) | (abs(X['area'] - X['area_min']) <= abs(X['area'] - X['area_median'])), True, False)
    X['nearest_area_class_max'] = np.where(( abs(X['area'] - X['area_max']) <= abs(X['area'] - X['area_min'])) | (abs(X['area'] - X['area_max']) <= abs(X['area'] - X['area_median'])), True, False)
    return X