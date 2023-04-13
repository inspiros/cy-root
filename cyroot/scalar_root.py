from inspect import getmembers
from functools import partial

from . import scalar_bracketing, scalar_quasi_newton, scalar_newton
from .utils.function_tagging import is_tagged_with_any_startswith

__all__ = [
    'SCALAR_ROOT_FINDING_METHODS',
    'find_root_scalar',
]

# noinspection DuplicatedCode
SCALAR_ROOT_FINDING_METHODS = {}
for module in [scalar_bracketing, scalar_quasi_newton, scalar_newton]:
    SCALAR_ROOT_FINDING_METHODS.update(
        getmembers(module, partial(is_tagged_with_any_startswith, start='cyroot.scalar')))


# noinspection DuplicatedCode
def find_root_scalar(method: str, *args, **kwargs):
    """
    Find the root of a scalar function.

    Args:
        method (str): Name of the method. A full list of supported
         methods is stored in ``SCALAR_ROOT_FINDING_METHODS``.
        *args: Extra argument to be passed.
        **kwargs: Extra keywords arguments to be passed.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    if method in SCALAR_ROOT_FINDING_METHODS.keys():
        return SCALAR_ROOT_FINDING_METHODS[method](*args, **kwargs)
    else:
        raise ValueError(f'No implementation for {str(method)} found. '
                         f'Supported methods are: {", ".join(SCALAR_ROOT_FINDING_METHODS.keys())}')
