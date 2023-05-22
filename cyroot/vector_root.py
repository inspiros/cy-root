from inspect import getmembers
from functools import partial
from . import vector_bracketing, vector_quasi_newton, vector_newton
from .utils._function_registering import is_tagged_with_any_startswith

__all__ = [
    'VECTOR_ROOT_FINDING_METHODS',
    'find_vector_root',
]

# noinspection DuplicatedCode
VECTOR_ROOT_FINDING_METHODS = {}
for module in [vector_bracketing, vector_quasi_newton, vector_newton]:
    VECTOR_ROOT_FINDING_METHODS.update(
        getmembers(module, partial(is_tagged_with_any_startswith, start='cyroot.vector')))


# noinspection DuplicatedCode
def find_vector_root(method: str, *args, **kwargs):
    """
    Find the root of a vector function.

    Args:
        method (str): Name of the method. A full list of supported
         methods is stored in ``VECTOR_ROOT_FINDING_METHODS``.
        *args: Extra arguments to be passed.
        **kwargs: Extra keyword arguments to be passed.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    if method in VECTOR_ROOT_FINDING_METHODS.keys():
        return VECTOR_ROOT_FINDING_METHODS[method](*args, **kwargs)
    elif 'generalized_' + method in VECTOR_ROOT_FINDING_METHODS.keys():
        return VECTOR_ROOT_FINDING_METHODS['generalized_' + method](*args, **kwargs)
    else:
        raise ValueError(f'No implementation for {method} found. '
                         f'Supported methods are: {", ".join(VECTOR_ROOT_FINDING_METHODS.keys())}')
