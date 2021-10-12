from .objective_base import ObjectiveBase

from bciavm.problem_types import ProblemTypes


class RegressionObjective(ObjectiveBase):
    """Base class for all regression objectives."""

    problem_types = [ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]
