import warnings

__all__ = [
    'ValueWarning',
    'warn_value',
]


class ValueWarning(UserWarning):
    pass


warnings.simplefilter('always', ValueWarning)


def warn_value(message, filename=None, lineno=None):
    if filename is not None and lineno is not None:
        warnings.warn_explicit(message, ValueWarning, filename, lineno, module=None)
    else:
        warnings.warn(message, ValueWarning, stacklevel=2)
