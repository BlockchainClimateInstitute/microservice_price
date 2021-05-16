import warnings
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

from woodwork.logical_types import Boolean, Integer, Double, Categorical
from woodwork.logical_types import NaturalLanguage as String

# hack to prevent warnings from skopt
# must import sklearn first
import sklearn
import bciavm.model_family
import bciavm.objectives
import bciavm.pipelines
import bciavm.preprocessing
import bciavm.problem_types
import bciavm.utils
import bciavm.data
import bciavm.core

from bciavm.utils import print_info
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    warnings.simplefilter("ignore", DeprecationWarning)
    import skopt
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', 'The following selectors were not present in your DataTable')

__version__ = '1.21.1'
