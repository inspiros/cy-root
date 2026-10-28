import warnings

__all__ = [
    'ValueWarning', 'NotImplementedWarning',
    'warn', 'warn_value', 'warn_not_implemented',
    'set_warning_filter',
]


class ValueWarning(UserWarning):
    pass


class NotImplementedWarning(UserWarning):
    pass


warnings.simplefilter('always', ValueWarning)
warnings.simplefilter('always', NotImplementedWarning)


def warn(message, category=UserWarning, stacklevel=2, filename=None, lineno=None):
    if filename is not None and lineno is not None:
        warnings.warn_explicit(message, category, filename, lineno, module=None)
    else:
        warnings.warn(message, category, stacklevel=stacklevel)


def warn_value(message, filename=None, lineno=None):
    warn(message, ValueWarning, 3, filename, lineno)


def warn_not_implemented(message, filename=None, lineno=None):
    warn(message, NotImplementedWarning, 3, filename, lineno)


def set_warning_filter(action: str = 'always',
                       category: Warning = UserWarning,
                       lineno: int = 0,
                       append: bool = False):
    """
    Add warning filter.

    Args:
        action (str): one of ``'error'``, ``'ignore'``, ``'always'``,
         ``'default'``, ``'module'``, or ``'once'``. Defaults to ``'always'``.
        category (Warning): the category of the warning. Defaults to ``UserWarning``.
        lineno (int): an integer line number, 0 matches all warnings.
         Defaults to 0.
        append (bool): if True, append to the list of filters. Defaults to False.

    See Also:
        ``warnings.simplefilter``
    """
    warnings.simplefilter(action, category, lineno, append)
