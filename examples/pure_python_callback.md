# Compile Python Syntax Into A Native Callback With Cython

Cython's pure Python mode lets a `.py` source file describe C functions using
annotations and decorators. This example compiles `x*x - 2` and its derivative,
then hands native function pointers to `cyroot`. The solve executes without the
GIL when both callbacks are native.

This is ahead-of-time compilation. It requires Cython and a C++ compiler; it does
not compile an arbitrary Python callable on its first invocation as Numba's JIT
can. Python objects, NumPy calls, closures over Python state, and arbitrary Python
library calls cannot run inside these `nogil` functions. Use typed C values and
Cython's C-level imports for such operations as mathematical functions.

With `cyroot`, Cython 3, and setuptools installed in the active environment, copy
`pure_python_callback.py` and `pure_python_callback_setup.py` to a working
directory, then run:

```sh
python pure_python_callback_setup.py build_ext --inplace
python -c 'import pure_python_callback as e; answer, history = e.solve(); print(answer)'
```

The source uses valid Python syntax, but this callback module must be compiled
before importing: `cython.cimports` resolves the native bridge during compilation.
The extension produced beside the `.py` file takes precedence on import.

From Python:

```python
from cyroot.native import Newton
import pure_python_callback as example

assert example.residual(2.0) == 2.0
assert example.callback().native
solver = Newton(example.callback(), example.derivative_callback(), 1.5, history=32)
answer = solver.run()
assert answer.success
print(answer.root)  # approximately sqrt(2)
```

`@cython.ccall` gives each typed numerical function a Python wrapper and a native
C entry point. Passing `example.residual` directly to a solver uses that Python
wrapper and acquires the GIL for calls. `example.callback()` instead returns a
`ScalarCallback` with an ABI bridge that calls the C entry point directly.
The bridge writes the result to `out[0]` and returns status zero. The derivative
uses the same convention.

`@cython.nogil` allows a C caller to invoke these functions without the GIL; it
does not itself release the GIL. The native solver releases it around the loop.
`@cython.exceptval(check=False)` declares the functions non-throwing, avoiding
Python exception checks in that loop. This is appropriate for the arithmetic
shown here; do not use it to hide operations that need Python exceptions. A
fallible native bridge should check its conditions and return a nonzero status
for `cyroot` to report as a callback failure.

These callbacks are stateless: their context pointers are null. The descriptors
retain the Python wrappers as keepalives. For parameters or external buffers,
use an owner that keeps the native context alive for the entire callback
lifetime, as in [`native_scalar.pyx`](native_scalar.pyx). Do not retain pointers
to temporary or Python-managed objects that can be resized during a solve.

No package-level JIT decorator or runtime compilation dependency is introduced
by this example. The same ahead-of-time approach can wrap the vector and tensor
callback ABIs with typed output pointers.

## Vector Functions In The Same `.py` Module

The example also exports `vector_callback()` and `vector_jacobian_callback()`.
They describe three equations in two variables, with root `(1, 2)`:

```python
import numpy as np

def numpy_style(x):
    return np.array([x[0]**2 + x[1] - 3,
                     2*x[0] + x[1]**2 - 6,
                     x[0] + x[1] - 3])
```

That ordinary Python implementation remains supported. For GIL-free execution,
the compiled version uses the native callback signature and fills `out` directly:

```cython
@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def _vector_residual(
    x: cython.pointer(cython.const[cython.double]),
    n: cython.size_t,
    m: cython.size_t,
    out: cython.pointer(cython.double),
    context: cython.p_void,
) -> cython.int:
    if n != 2 or m != 3:
        return 1
    out[0] = x[0]*x[0] + x[1] - 3.0
    out[1] = 2.0*x[0] + x[1]*x[1] - 6.0
    out[2] = x[0] + x[1] - 3.0
    return 0
```

Its factory uses
`make_rectangular_callback` from `cython.cimports.cyroot.native.rectangular`,
declares dimensions `(2, 3)` and pins an owner. This stateless example pins the
factory function. The solver supplies valid input and output storage; the
callback creates no arrays. All `m` output elements must be written on success.
Do not modify or retain the borrowed input pointer.

Build with the same command as above, then run:

```python
import pure_python_callback as example

answer, history = example.solve_vector()
assert answer.success
print(answer.root)  # approximately (1, 2)

# Omit the analytic Jacobian: native finite differences sample the same callback.
answer, history = example.solve_vector(numerical_jacobian=True)
assert answer.success
```

The analytic Jacobian writes a 3-by-2 matrix into the supplied output buffer in
column-major order: `out[row + column*3]`. The same pattern works for square
systems with matching dimensions.

Cython does not generally turn `np.array`, `np.sin`, `np.dot` or `np.linalg.solve`
into GIL-free C operations. Replace allocation with output writes, elementwise
expressions with typed loops/C math, and suitable linear algebra with native
BLAS/LAPACK calls. NumPy arrays can still supply data at the Python boundary;
creating a typed memoryview there once does not require the GIL for every later
typed element access. Our callback ABI uses pointers to avoid constructing a
new Python view for each residual evaluation. Avoid assuming that compiling an
unchanged NumPy function removes its Python calls or improves its speed.
