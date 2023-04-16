from dynamic_default_args import dynamic_default_args, named_default
import numpy as np

from ._check_args import _check_stop_cond_args

__all__ = [
    'set_default_stop_condition_args',
]

# stop condition args
ETOL = 1e-10
ERTOL = 4 * np.finfo(np.float64).eps
PTOL = 1e-12
PRTOL = 4 * np.finfo(np.float64).eps
MAX_ITER = 200

# derivative approximation args
FINITE_DIFF_STEP = 1e-3


@dynamic_default_args()
def set_default_stop_condition_args(etol=named_default(ETOL=ETOL),
                                    ertol=named_default(ERTOL=ERTOL),
                                    ptol=named_default(PTOL=PTOL),
                                    prtol=named_default(PRTOL=PRTOL),
                                    max_iter=named_default(MAX_ITER=MAX_ITER)):
    """
    Check default values for etol, ertol, ptol, prtol, and max_iter.
    This function uses default values to be modified as its own inputs,
    so None value will be interpreted as disabling the stop condition (set to 0).

    Args:
        etol (float, optional): Error tolerance, indicating the
         desired precision of the root. Defaults to {etol}.
        ertol (float, optional): Relative error tolerance.
         Defaults to {ertol}.
        ptol (float, optional): Precision tolerance, indicating
         the minimum change of root approximations or width of
         brackets (in bracketing methods) after each iteration.
         Defaults to {ptol}.
        prtol (float, optional): Relative precision tolerance.
         Defaults to {prtol}.
        max_iter (int, optional): Maximum number of iterations.
         If set to 0, the procedure will run indefinitely until
         stopping condition is met. Defaults to {max_iter}.
    """
    etol, ertol, ptol, prtol, max_iter = _check_stop_cond_args(etol,
                                                               ertol,
                                                               ptol,
                                                               prtol,
                                                               max_iter)
    named_default('ETOL').value = etol
    named_default('ERTOL').value = ertol
    named_default('PTOL').value = ptol
    named_default('PRTOL').value = prtol
    named_default('MAX_ITER').value = max_iter
