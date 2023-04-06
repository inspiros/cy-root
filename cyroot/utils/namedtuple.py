"""
namedtuple with defaults for python versions lower than 3.7
"""

import sys
from collections import namedtuple as _namedtuple

__all__ = ['namedtuple']


if sys.version_info[:2] >= (3, 7):
    namedtuple = _namedtuple
else:
    def namedtuple(typename, field_names, *, rename=False, defaults=None, module=None):
        res = _namedtuple(typename, field_names, rename=rename, module=module)
        if defaults is not None:
            res.__new__.__defaults__ = tuple(defaults)
        return res
