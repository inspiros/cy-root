from inspect import getmembers

from . import newton, quasi_newton, bracketing
from .utils.function_tagging import has_tag

__all__ = [
    'SCALAR_ROOT_FINDING_METHODS',
    'find_root_scalar',
]

SCALAR_ROOT_FINDING_METHODS = {}
for module in [bracketing, newton, quasi_newton]:
    SCALAR_ROOT_FINDING_METHODS.update(getmembers(module, has_tag))


def find_root_scalar(method: str, *args, **kwargs):
    """
    Find the root of a scalar function.

    Args:
        method: Name of the method. A full list of supported
         methods is stored in ``SCALAR_ROOT_FINDING_METHODS``.
        *args: Extra argument to be passed.
        **kwargs: Extra keywords arguments to be passed.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    if method not in SCALAR_ROOT_FINDING_METHODS.keys():
        raise ValueError(f'No implementation for {str(method)} found. '
                         f'Supported methods are: {", ".join(SCALAR_ROOT_FINDING_METHODS.keys())}')
    return SCALAR_ROOT_FINDING_METHODS[method](*args, **kwargs)
