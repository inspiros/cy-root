# Building And Testing

The development sources support CPython 3.11–3.14 and require a C++17 compiler
and Python development headers. Linux distributions often provide those headers
in a separate `python3-dev` or version-specific development package.

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

The test runner defaults to at most four concurrent processes, or the CPU count
if smaller. It discovers test modules without importing them in the coordinator,
then runs each module once in a fresh process using the current Python
interpreter. External Cython consumer builds keep their separate temporary
directories. `--jobs 1` retains process isolation while running serially:

```sh
python tests/run_tests.py --jobs 1
python tests/run_tests.py --jobs 4 --pattern test_native_broyden.py
python tests/run_tests.py --self-check
```

The default discovery directory is the runner's own `tests` directory, so an
absolute invocation also works outside the checkout. `--start-directory` and
`--pattern` select another suite. Verbose per-module logs and machine-readable
results are written to `test-logs/` in the current directory; use
`--log-directory` to choose another location. `summary.json` contains aggregate
counts and the executed test IDs for every module. Test failures, import or
discovery errors, unexpected successes and worker crashes produce a nonzero
exit status. Optional `--expected-tests N` checks the aggregate count;
`--require-no-skips` rejects skips when the environment supplies every compiler
and optional test dependency.

Keep new tests focused on distinct behavior. Test numerical identities, failure
states and buffer safety in the native engine's suite. Wrapper tests should
check argument forwarding, dispatch and the Python interface, without repeating
the engine's numerical cases for every alias. Keep regressions for bugs found
in practice and compiled-consumer checks for the native API. For a local change,
run the affected module first; run the full suite once after integration.

The runner preserves `CC`, `CXX` and other caller environment settings. It sets
unset OpenMP, OpenBLAS, MKL, Accelerate, NumExpr and BLIS thread limits to one to
avoid multiplying numerical backend threads across test workers. Explicit
thread settings remain in effect.

NumPy and SciPy remain runtime dependencies. SciPy supplies the Cython
LAPACK interface; no separate system BLAS/LAPACK installation is required by
this backend. SciPy is currently restricted to versions below 1.18 because the
newer Cython BLAS/LAPACK integer ABI has not been validated by this package.

Source builds use Cython 3.3.0, declared in `pyproject.toml`. Authored `.pyx` and
`.pxd` files are authoritative. Generated C++ is placed in `build/cython/`;
stale generated files next to `.pyx` files have been removed. Source archives
require Cython to build and do not bundle generated C++.
Installed packages include Cython declarations and native headers for compiled
consumers.

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

Use the same process for the source archive. Source installation needs Cython,
Python development headers, and the compiler. Test discovery is limited to
`tests/`; `develop/` contains private or disposable experiments and `examples/`
contains demonstrations.

`--installed-prefix` checks the imported `cyroot` package and every loaded
`cyroot` module in each worker before and after its tests. A source import
outside that prefix fails the run. The runner adds the test discovery directory
for sibling test helpers; it never adds the checkout root. Source testing
therefore requires the editable installation shown above. For installed-wheel
verification, keep the working directory outside the checkout and supply the
fresh environment's prefix.

The native callback tests compile a small external Cython extension against
the installed package, so their test environment needs Cython and a compiler.
The standalone native header check uses `g++` when available. On Linux, run
`CYROOT_NATIVE_SANITIZERS=1 python tests/run_tests.py --pattern test_native_buffers.py --jobs 1`
for address/undefined-behavior checks. The dedicated CI job additionally enables
leak detection; traced local environments may not support LeakSanitizer.
