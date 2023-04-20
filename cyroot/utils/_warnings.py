import warnings

__all__ = [
    'ValueWarning',
    'warn_value',
    'set_value_warning_filter',
]


class ValueWarning(UserWarning):
    pass


warnings.simplefilter('always', ValueWarning)


def warn_value(message, filename=None, lineno=None):
    if filename is not None and lineno is not None:
        warnings.warn_explicit(message, ValueWarning, filename, lineno, module=None)
    else:
        warnings.warn(message, ValueWarning, stacklevel=2)


def set_value_warning_filter(action: str = 'always',
                             lineno: int = 0,
                             append: bool = False):
    """
    Add value warning filter.

    Args:
        action (str): one of ``'error'``, ``'ignore'``, ``'always'``,
         ``'default'``, ``'module'``, or ``'once'``. Defaults to ``'always'``.
        lineno (int): an integer line number, 0 matches all warnings.
         Defaults to 0.
        append (bool): if True, append to the list of filters. Defaults to False.

    See Also:
        ``warnings.simplefilter``
    """
    warnings.simplefilter(action, ValueWarning, lineno, append)
