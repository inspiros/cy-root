"""Build native_least_squares.pyx against installed cy-root declarations."""
from pathlib import Path
import sys
from Cython.Build import cythonize
from setuptools import Extension, setup
from cyroot.native import get_include, get_cython_include

flags = ['/std:c++17'] if sys.platform == 'win32' else ['-std=c++17']
source = Path(__file__).with_name('native_least_squares.pyx')
setup(name='cyroot-native-least-squares-example', ext_modules=cythonize([
    Extension('native_least_squares', [str(source)], language='c++',
              include_dirs=[get_include()], extra_compile_args=flags)],
    language_level=3, include_path=[get_cython_include()], build_dir='build/cython'))
