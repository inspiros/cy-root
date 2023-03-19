"""
namedtuple with defaults for python versions lower than 3.7
"""

import sys
from collections import namedtuple as _namedtuple

__all__ = ['namedtuple']


def namedtuple(typename, field_names, *, rename=False, defaults=None, module=None):
    if sys.version_info[:2] >= (3, 7):
        result = _namedtuple(typename, field_names, rename=rename, defaults=defaults, module=module)
    else:
        result = _namedtuple(typename, field_names, rename=rename, module=module)
        if defaults is not None:
            result.__new__.__defaults__ = tuple(defaults)
    return result
