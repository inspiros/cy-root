import math

__all__ = [
    '_check_stopping_condition_args',
    '_check_bracket',
    '_check_bracket_val',
]


def _check_stopping_condition_args(etol: float, ptol: float, max_iter: int):
    if etol < 0:
        raise ValueError(f'etol must be positive. Got {etol}.')
    if ptol < 0:
        raise ValueError(f'ptol must be positive. Got {ptol}.')
    if not isinstance(max_iter, int) or max_iter == float('inf'):
        raise ValueError(f'max_iter must be an integer. Got {max_iter}.')
    if etol == ptol == 0 and max_iter <= 0:
        raise ValueError(f'Disabling both etol, ptol, and max_iter will '
                         f'likely cause the algorithm to run indefinitely.')


def _check_bracket(a: float, b: float, check_nan=True):
    if check_nan and (math.isnan(a) or math.isnan(b)):
        raise ValueError(f'nan value encountered a={a}, b={b}.')
    if a > b:
        raise ValueError(f'Expect a<b. Got a={a}, b={b}.')


def _check_bracket_val(f_a: float, f_b: float, check_nan=True):
    if check_nan and (math.isnan(f_a) or math.isnan(f_b)):
        raise ValueError(f'nan value encountered a={f_a}, b={f_b}.')
    if math.copysign(1, f_a) == math.copysign(1, f_b):
        raise ValueError('f_a and f_b must have opposite sign. '
                         f'Got f_a={f_a} and f_b={f_b}.')
