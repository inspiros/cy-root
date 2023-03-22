__all__ = [
    'VECTOR_ROOT_FINDING_METHODS',
    'find_root_vector',
]

VECTOR_ROOT_FINDING_METHODS = {}


def find_root_vector(method: str, *args, **kwargs):
    """
    Find the root of a vector function.

    Args:
        method: Name of the method. A full list of supported
         methods is stored in ``VECTOR_ROOT_FINDING_METHODS``.
        *args: Extra argument to be passed.
        **kwargs: Extra keywords arguments to be passed.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    raise NotImplementedError('Vector root is not supported yet.')
