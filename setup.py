import codecs
import os

import _version
from setuptools import Extension, setup, find_packages

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True


def get_ext_modules():
    import numpy
    # Find all includes
    package_root = 'cyroot'
    include_dirs = [
        package_root,
        numpy.get_include(),
    ]

    ext_modules = []
    ext = '.pyx' if use_cython else '.c'
    for root, dirs, files in os.walk(package_root):
        for f in filter(lambda f: f.endswith(ext), files):
            f_path = os.path.join(root, f)
            ext_modules.append(
                Extension(name=os.path.splitext(f_path)[0].replace(os.sep, '.'),
                          sources=[f_path],
                          include_dirs=include_dirs)
            )
    if use_cython:
        ext_modules = cythonize(ext_modules,
                                language_level='3')
    # Set up the ext_modules for Cython or not, depending
    return ext_modules


def setup_package():
    setup(
        name='cy-root',
        version=_version.__version__,
        description='A Cython implementation of multiple root-finding methods.',
        long_description=codecs.open('README.md', mode='r', encoding='utf-8').read(),
        url='https://github.com/inspiros/cy-root',
        author='Hoang-Nhat Tran (inspiros)',
        author_email='hnhat.tran@gmail.com',
        license='MIT',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Cython',
            'Topic :: Scientific/Engineering :: Mathematics',
        ],
        keywords='root-finding',
        project_urls={
            'Source': 'https://github.com/inspiros/cy-root',
        },
        python_requires='>=3.8',
        install_requires=[
            'numpy',
            'sympy',
        ],
        extras_require={
            'dev': [
                'cython',
                'pyximport',
            ]
        },
        setup_requires=[
            'numpy',
            'sympy',
        ],
        packages=find_packages(),
        ext_modules=get_ext_modules(),
        cmdclass={'build_ext': build_ext} if use_cython else {},
    )


if __name__ == '__main__':
    setup_package()
