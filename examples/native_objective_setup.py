"""Run inside examples: python native_objective_setup.py build_ext --inplace."""
import sys
from setuptools import Extension, setup
from Cython.Build import cythonize
from cyroot.native import get_include, get_cython_include

setup(ext_modules=cythonize([Extension('native_objective',['native_objective.pyx'],
    language='c++',include_dirs=[get_include()],
    extra_compile_args=['/std:c++17'] if sys.platform=='win32' else ['-std=c++17'])],
    language_level=3,include_path=[get_cython_include()]))
