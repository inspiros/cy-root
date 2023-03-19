"""
For importing cython extensions without running ``setup.py``.
Intended to be used only during development.
"""

import os

import numpy as np
import pyximport

__all__ = [
    '_has_ext'
]

try:
    include_dirs = [
        np.get_include(),
        os.path.dirname(__file__),
    ]
    for root, dirs, files in os.walk(os.path.dirname(__file__)):
        for dir in filter(lambda _: _ != '__pycache__', dirs):
            include_dirs.append(os.path.join(root, dir))

    pyximport.install(
        setup_args=dict(
            include_dirs=include_dirs,
            script_args=["--cython-cplus"]
        ),
        inplace=True,
        language_level='3')


    def _has_ext():
        return True
except Exception as e:
    print('Unable to compile Cython extension.', e)


    def _has_ext():
        return False
