# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

import math
import operator


__all__ = [
    '_check_stop_cond_args',
]

################################################################################
# Python-side checks
################################################################################
# noinspection DuplicatedCode
def _check_stop_cond_arg(tol: float, arg_name='tol'):
    if tol is None:
        return 0
    elif math.isnan(tol) or not math.isfinite(tol) or tol < 0:
        raise ValueError(f'{arg_name} must be non-negative finite number. Got {tol}.')
    return tol

# noinspection DuplicatedCode
def _check_stop_cond_args(etol: float,
                          ertol: float,
                          ptol: float,
                          prtol: float,
                          max_iter):
    r"""Check tolerances and max_iter."""
    etol = _check_stop_cond_arg(etol, 'etol')
    ertol = _check_stop_cond_arg(ertol, 'ertol')
    ptol = _check_stop_cond_arg(ptol, 'ptol')
    prtol = _check_stop_cond_arg(prtol, 'prtol')
    if ertol >= 1 or prtol >= 1:
        raise ValueError('relative tolerances must be smaller than one.')

    if max_iter is None:
        max_iter = 0
    else:
        if isinstance(max_iter, bool):
            raise ValueError('max_iter must be a non-negative integer, not bool.')
        try:
            max_iter = operator.index(max_iter)
        except TypeError:
            raise ValueError(f'max_iter must be a non-negative integer. Got {max_iter}.') from None
        if max_iter < 0 or max_iter > <size_t> -1:
            raise ValueError('max_iter must fit a non-negative size_t.')

    if etol == ertol == ptol == prtol == max_iter == 0:
        raise ValueError(f'Disabling both tolerances and max_iter will '
                         f'likely cause the algorithm to run indefinitely.')
    return etol, ertol, ptol, prtol, max_iter
