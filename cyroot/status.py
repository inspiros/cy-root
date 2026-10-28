"""Stable reasons for solver and interval-search termination."""
from enum import StrEnum


class TerminationReason(StrEnum):
    RESIDUAL_TOLERANCE = "residual_tolerance"
    BRACKET_TOLERANCE = "bracket_tolerance"
    STAGNATION = "stagnation"
    MAX_ITERATIONS = "max_iterations"
    ZERO_DERIVATIVE = "zero_derivative"
    BREAKDOWN = "breakdown"
    NONFINITE = "nonfinite"
    MAX_SPLIT_ITERATIONS = "max_split_iterations"
    COMPLETED = "completed"
    UNSPECIFIED = "unspecified"
    CALLBACK_FAILURE = "callback_failure"
    STATIONARY_NONROOT = "stationary_nonroot"
    MAX_EVALUATIONS = "max_evaluations"
    NO_ACCEPTABLE_STEP = "no_acceptable_step"
    LINEAR_FAILURE = "linear_failure"
    DERIVATIVE_BREAKDOWN = "derivative_breakdown"
    LINE_SEARCH_FAILURE = "line_search_failure"
    OPTIMIZER_STOPPED_WITHOUT_ROOT = "optimizer_stopped_without_root"


__all__ = ["TerminationReason"]
