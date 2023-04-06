from inspect import getmembers

from . import vector_quasi_newton, vector_newton
from .utils.function_tagging import has_tag

__all__ = [
    'VECTOR_ROOT_FINDING_METHODS',
    'find_root_vector',
]

VECTOR_ROOT_FINDING_METHODS = {}
for module in [vector_quasi_newton, vector_newton]:
    VECTOR_ROOT_FINDING_METHODS.update(getmembers(module, has_tag))


def find_root_vector(method: str, *args, **kwargs):
    """
    Find the root of a vector function.

    Args:
        method (str): Name of the method. A full list of supported
         methods is stored in ``VECTOR_ROOT_FINDING_METHODS``.
        *args: Extra argument to be passed.
        **kwargs: Extra keywords arguments to be passed.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    if method not in VECTOR_ROOT_FINDING_METHODS.keys():
        raise ValueError(f'No implementation for {str(method)} found. '
                         f'Supported methods are: {", ".join(VECTOR_ROOT_FINDING_METHODS.keys())}')
    return VECTOR_ROOT_FINDING_METHODS[method](*args, **kwargs)
