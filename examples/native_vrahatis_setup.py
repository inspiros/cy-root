"""Build the native Vrahatis callback example against an installed cyroot."""
from pathlib import Path
import sys
from setuptools import Extension,setup
from Cython.Build import cythonize
from cyroot.native import get_include,get_cython_include

flags=['/std:c++17'] if sys.platform=='win32' else ['-std=c++17']
setup(ext_modules=cythonize([Extension('native_vrahatis',
    [str(Path(__file__).with_name('native_vrahatis.pyx'))],language='c++',
    include_dirs=[get_include()],extra_compile_args=flags)],language_level=3,
    include_path=[get_cython_include()],build_dir='build/cython'))
