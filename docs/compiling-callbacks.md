# Compile Python-Syntax Callbacks With Cython

Yes: users can write typed numerical functions in a `.py` file, compile them
with Cython's **pure Python mode**, and supply native callbacks to cy-root.
The [complete working example](../examples/pure_python_callback.md) includes the
source and build command. It uses existing callback interfaces; no package JIT
or new runtime dependency is required.

The numerical part looks like Python with explicit C types:

```python
import cython

@cython.ccall
@cython.nogil
@cython.exceptval(check=False)
def residual(x: cython.double) -> cython.double:
    return x*x - 2.0
```

Cython generates both a Python-callable wrapper and a C entry point. The example
adds a small native bridge from that C entry point to cy-root's status/output
callback signature, then exports `callback()` and `derivative_callback()`:

```python
from cyroot.native import Newton
import pure_python_callback as example  # the compiled extension

answer = Newton(example.callback(), example.derivative_callback(), 1.5).run()
assert answer.success
```

Passing `example.residual` directly is also valid as a Python callback, but
re-enters Python's call machinery and requires the GIL. The descriptor from
`example.callback()` selects the native C entry point. `nogil` permits native
calls without the GIL; the solver releases it around numerical execution.
The stateless example pins its function wrappers as owners and has null context
pointers. Parameterized functions need a pinned native context owner.

## How This Differs From Numba

The recommended Cython route is ahead-of-time: Cython translates the source into
C/C++, and a compiler builds an importable extension. Type annotations identify
C arithmetic. A `.py` file alone, imported as ordinary Python, does not become
native code. The example's `cython.cimports` also requires compilation before
import. Neither a `.py` nor a `.pyx` suffix alone determines GIL behavior.

Cython also supplies runtime compilation facilities such as `cython.inline`
and `@cython.compile`. These can compile suitable source/specializations, but
require a build toolchain and have source, typing and Python-state constraints.
They produce Python-callable functions; they do not automatically export
cy-root's callback ABI or establish a no-GIL contract. They are therefore not
a drop-in `numba.njit` equivalent for our fully native solver path. A runtime
helper could eventually automate compilation and bridge generation for a
restricted typed subset, but this package does not currently expose one.

## Native Restrictions

To stay outside the GIL, the body must use native types and operations. Arbitrary
Python objects, Python-level NumPy calls, dynamic Python closures and most Python
library calls require it. Native C math imports can replace many scalar `math`
operations; vector functions can write into the existing callback's output span
using typed pointers. Numerical derivatives can then sample the native residual
without returning to Python.

`exceptval(check=False)` is suitable only when exceptions cannot escape.
A fallible bridge should validate its own conditions and return a nonzero status,
which cy-root reports through `NativeCallbackError`. Shared mutable native
context must support the intended concurrency. Compilation does not prove
thread safety, accuracy or speed.

For raw scalar, rectangular vector and Hessian interfaces, see the existing
[native core guide](native-core.md), [rectangular callbacks](native-rectangular-callbacks.md)
and [tensor callbacks](native-rectangular-tensors.md). The pure-Python example
needs no NumPy array or NumPy C headers.

## Vector Functions And NumPy

The same [pure-Python example](../examples/pure_python_callback.md#vector-functions-in-the-same-py-module)
includes a compiled vector residual, analytic Jacobian and native numerical
Jacobian solve. It accepts two variables and writes three residuals directly
into solver-owned output storage. No ndarray is created per sample.

Compiling an existing NumPy function does not generally remove its Python calls.
For the native path, translate elementwise expressions into typed operations or
loops, use C math functions, and call native BLAS/LAPACK for applicable matrix
operations. NumPy arrays remain useful at setup/export boundaries. Typed
memoryviews can also be acquired once at a boundary and accessed without the GIL;
creating a Python view on each callback invocation is unnecessary with our raw
pointer interface. Arbitrary NumPy/SciPy routines still require their own native
implementation or a Python callback bridge.
