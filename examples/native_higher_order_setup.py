"""Compile native_higher_order.pyx against an installed cy-root package."""
from pathlib import Path
import sys

from Cython.Build import cythonize
from setuptools import Extension, setup
from cyroot.native import get_include, get_cython_include

source = Path(__file__).with_name("native_higher_order.pyx")
flags = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]
setup(
    name="cyroot-native-higher-order-example",
    ext_modules=cythonize(
        [Extension("native_higher_order", [str(source)], language="c++",
                   include_dirs=[get_include()], extra_compile_args=flags)],
        language_level=3,
        include_path=[get_cython_include()],
        build_dir="build/cython",
    ),
)
