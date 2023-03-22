from ._check_args import _check_stop_condition_args
from .utils.dynamic_default_args import dynamic_default_args, named_default

__all__ = [
    'set_default_stop_condition_args',
]

ETOL = 1e-10
PTOL = 1e-12
MAX_ITER = 1000


@dynamic_default_args()
def set_default_stop_condition_args(etol=named_default(ETOL=ETOL),
                                    ptol=named_default(PTOL=PTOL),
                                    max_iter=named_default(MAX_ITER=MAX_ITER)):
    """
    Check default values for etol, ptol, and max_iter. This function uses default
    values to be modified as its own inputs, so None value will be interpreted as
    disabling the stop condition (set to 0).

    Args:
        etol: Error tolerance, indicating the desired precision
         of the root. Dynamically defaults to {etol}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Dynamically defaults to {ptol}.
        max_iter: Maximum number of iterations. If set to 0, the
         procedure will run indefinitely until stopping condition is
         met. Dynamically defaults to {max_iter}.
    """
    etol, ptol, max_iter = _check_stop_condition_args(etol, ptol, max_iter)
    named_default('ETOL').value = etol
    named_default('PTOL').value = ptol
    named_default('MAX_ITER').value = max_iter
