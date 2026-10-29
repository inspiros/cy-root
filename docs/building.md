# Building And Testing

CyRoot supports CPython 3.11–3.14. Building from source needs a C++17 compiler
and the Python development headers, which Linux distributions often ship in a
separate `python3-dev` (or version-specific) package.

Create an isolated environment, then build the package in editable mode:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python tests/run_tests.py
```

On Windows, activate with `.venv\Scripts\activate` and use an MSVC toolchain with
C++17 support. On macOS, install the Xcode command line tools.

The test runner uses up to four worker processes, or fewer on machines with
fewer CPUs. It discovers test modules without importing them, then runs each
module once in a fresh process of the current interpreter. External Cython
consumer builds use their own temporary directories. `--jobs 1` runs the
modules one at a time, still in separate processes:

```sh
python tests/run_tests.py --jobs 1
python tests/run_tests.py --jobs 4 --pattern test_native_broyden.py
python tests/run_tests.py --self-check
```

By default the runner discovers tests in its own `tests` directory, so it also
works when invoked by absolute path from outside the checkout.
`--start-directory` and `--pattern` select a different set. Per-module logs and
machine-readable results go to `test-logs/` in the current directory, or to
`--log-directory`; `summary.json` holds the totals and the test IDs run in each
module. Test failures, import or discovery errors, unexpected successes and
worker crashes give a nonzero exit status. `--expected-tests N` also checks the
total count, and `--require-no-skips` fails on skips, for environments that
provide every compiler and optional test dependency.

When adding tests, cover distinct behavior. Numerical identities, failure
states and buffer safety belong in the native engine's suite. Wrapper tests
check argument forwarding, dispatch and the Python interface, without repeating
the engine's numerical cases for every alias. Keep regression tests for bugs
found in practice, and compiled-consumer checks for the native API. For a local
change, run the affected module first and the full suite before merging.

The runner passes `CC`, `CXX` and the rest of your environment through. Thread
limits for OpenMP, OpenBLAS, MKL, Accelerate, NumExpr and BLIS that you have
not set are set to one, so parallel workers do not multiply backend threads;
limits you set yourself are kept.

NumPy and SciPy are runtime dependencies. SciPy provides the Cython LAPACK
interface, so no separate system BLAS or LAPACK is needed. SciPy is limited to
versions below 1.18, whose new Cython BLAS/LAPACK integer ABI has not yet been
validated with CyRoot.

Source builds use Cython 3.3.0, as declared in `pyproject.toml`. Generated C++
goes to `build/cython/`. Source archives do not bundle generated C++, so
installing from one needs Cython. Installed packages include the Cython
declarations and native headers for compiled consumers.

Build distribution artifacts with:

```sh
python -m build
```

Test the wheel in a fresh environment, from outside the source directory, so
the checkout cannot mask a missing installed module:

```sh
python -m venv /tmp/cyroot-wheel-test
/tmp/cyroot-wheel-test/bin/python -m pip install dist/*.whl Cython==3.3.0 setuptools
cd /tmp
/tmp/cyroot-wheel-test/bin/python /absolute/path/to/cy-root/tests/run_tests.py \
  --jobs 4 --installed-prefix /tmp/cyroot-wheel-test \
  --log-directory /tmp/cyroot-wheel-test-logs
```

Test the source archive the same way; installing it needs Cython, the Python
development headers and a compiler. Only `tests/` is searched for tests;
`examples/` holds demonstrations.

With `--installed-prefix`, each worker checks before and after its tests that
`cyroot` and every loaded `cyroot` module come from that prefix, and fails the
run otherwise. The runner adds the test directory to the path for its helper
modules but never the checkout root, so testing the sources needs the editable
installation above. To verify an installed wheel, run from outside the checkout
and pass the fresh environment's prefix.

The native callback tests compile a small external Cython extension against
the installed package, so they need Cython and a compiler. The standalone
native header check uses `g++` when it is available. On Linux, run
`CYROOT_NATIVE_SANITIZERS=1 python tests/run_tests.py --pattern test_native_buffers.py --jobs 1`
for AddressSanitizer and UndefinedBehaviorSanitizer checks. The dedicated CI
job also enables leak detection; LeakSanitizer may not work in some sandboxed
or traced local environments.
