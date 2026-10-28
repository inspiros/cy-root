"""Build authored Cython sources; generated C++ stays under build/."""

import importlib.util
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup


EXTENSION_NAMES = (
    "cyroot._check_args",
    "cyroot.fptr",
    "cyroot.return_types",
    "cyroot.scalar_bracketing",
    "cyroot.scalar_derivative_approximation",
    "cyroot.scalar_newton",
    "cyroot.scalar_quasi_newton",
    "cyroot.vector_bracketing",
    "cyroot.vector_derivative_approximation",
    "cyroot.vector_newton",
    "cyroot.vector_opt",
    "cyroot.vector_quasi_newton",
    "cyroot.ops.scalar_types",
    "cyroot.ops.scalar_ops",
    "cyroot.ops.vector_ops",
    "cyroot.ops.matrix_ops",
    "cyroot.native.scalar",
    "cyroot.native._buffers",
    "cyroot.native.linalg",
    "cyroot.native.least_squares",
    "cyroot.native.rectangular",
    "cyroot.native.vector",
    "cyroot.native.vrahatis",
    "cyroot.native.polygons",
    "cyroot.native.derivatives",
    "cyroot.native.scalar_derivatives",
    "cyroot.native.secant",
    "cyroot.native.bracketing",
    "cyroot.native.interpolation",
    "cyroot.native.complex_callbacks",
    "cyroot.native.differentiation",
    "cyroot.native.homeier",
    "cyroot.native.hybisect",
    "cyroot.native.higher_order",
    "cyroot.native.higher_derivatives",
    "cyroot.native.higher_mixed",
    "cyroot.native.householder_solver",
    "cyroot.native.hessians",
    "cyroot.native.tensor_callbacks",
    "cyroot.native.vector_chebyshev",
    "cyroot.native.broyden",
    "cyroot.native.objective",
    "cyroot.native.objective_derivatives",
    "cyroot.native.levenberg_marquardt",
    "cyroot.native.bfgs",
)


def get_version():
    spec = importlib.util.spec_from_file_location("_version", "cyroot/_version.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__


def get_readme(version):
    """Pin README images (including single-URL picture sources) to the release tag."""
    base = f"https://raw.githubusercontent.com/inspiros/cy-root/v{version}/"

    def image_url(url):
        if not url or url.startswith(("/", "#")) or urlsplit(url).scheme:
            return url
        return base + url.removeprefix("./")

    text = Path("README.md").read_text(encoding="utf-8")
    text = re.sub(
        r"(<(?:img\b[^>]*?\ssrc|source\b[^>]*?\ssrcset)\s*=\s*)([\"'])([^\"']*)\2",
        lambda match: match[1] + match[2] + image_url(match[3]) + match[2],
        text, flags=re.IGNORECASE,
    )
    return re.sub(
        r"(!\[[^\]]*\]\(\s*)(<[^>]+>|[^\s)]+)(?=[\s)])",
        lambda match: match[1] + ("<" + image_url(match[2][1:-1]) + ">"
                                 if match[2].startswith("<") else image_url(match[2])),
        text,
    )


def get_ext_modules():
    compile_args = ["/std:c++17", "/O2"] if sys.platform == "win32" else ["-std=c++17", "-O3"]
    include_dirs = ["cyroot", "cyroot/ops", "cyroot/utils", "cyroot/native", numpy.get_include()]
    extensions = [
        Extension(
            name,
            sources=[str(Path(*name.split(".")).with_suffix(".pyx"))],
            include_dirs=include_dirs,
            language="c++",
            extra_compile_args=compile_args,
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
        for name in EXTENSION_NAMES
    ]
    return cythonize(extensions, language_level=3, build_dir="build/cython")


if __name__ == "__main__":
    version = get_version()
    setup(version=version, ext_modules=get_ext_modules(),
          long_description=get_readme(version), long_description_content_type="text/markdown")
