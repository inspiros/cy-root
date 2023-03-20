import os

from setuptools import Extension, setup

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

PACKAGE_ROOT = 'cyroot'


def get_version(version_file='_version.py'):
    import importlib.util
    version_file_path = os.path.abspath(os.path.join(PACKAGE_ROOT, version_file))
    spec = importlib.util.spec_from_file_location('_version', version_file_path)
    version_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(version_module)
    return str(version_module.__version__)


def get_ext_modules():
    import numpy
    # Find all includes
    include_dirs = [
        PACKAGE_ROOT,
        numpy.get_include(),
    ]

    ext_modules = []
    ext = '.pyx' if use_cython else '.cpp'
    for root, dirs, files in os.walk(PACKAGE_ROOT):
        for d in dirs:
            dir_path = os.path.join(root, d)
            if any(_.endswith(ext) for _ in os.listdir(dir_path)):
                include_dirs.append(dir_path)

    for root, dirs, files in os.walk(PACKAGE_ROOT):
        for f in filter(lambda f: f.endswith(ext), files):
            f_path = os.path.join(root, f)
            ext_modules.append(
                Extension(name=os.path.splitext(f_path)[0].replace(os.sep, '.'),
                          sources=[f_path],
                          include_dirs=include_dirs)
            )
    if use_cython:
        # Set up the ext_modules for Cython or not, depending
        ext_modules = cythonize(ext_modules, language_level='3')
    return ext_modules


def setup_package():
    setup(
        version=get_version(),
        ext_modules=get_ext_modules(),
        cmdclass={'build_ext': build_ext} if use_cython else {},
    )


if __name__ == '__main__':
    setup_package()
