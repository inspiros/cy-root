"""Build the pure-Python-mode callback example against an installed cyroot."""
from pathlib import Path
import sys

from Cython.Build import cythonize
from setuptools import Extension, setup
from cyroot.native import get_include, get_cython_include

source = Path(__file__).with_name("pure_python_callback.py")
flags = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]
setup(
    name="cyroot-pure-python-callback-example",
    ext_modules=cythonize(
        [Extension("pure_python_callback", [str(source)], language="c++",
                   include_dirs=[get_include()], extra_compile_args=flags)],
        language_level=3,
        include_path=[get_cython_include()],
        build_dir="build/cython",
    ),
)
