# Compile Python Callbacks With Cython

You can write typed numerical functions in an ordinary `.py` file, compile them
with Cython's **pure Python mode**, and pass them to CyRoot as native
callbacks. The [complete working example](../examples/pure_python_callback.md)
includes the source and the build command. It uses the standard callback
interfaces and needs no JIT or extra runtime dependency.

The numerical part looks like Python with explicit C types:

```python
import cython

@cython.ccall
@cython.nogil
@cython.exceptval(check=False)
def residual(x: cython.double) -> cython.double:
    return x*x - 2.0
```

Cython generates both a Python-callable wrapper and a C entry point. The
example adds a small native bridge from that C entry point to CyRoot's
status-and-output callback signature, and exports `callback()` and
`derivative_callback()`:

```python
from cyroot.native import Newton
import pure_python_callback as example  # the compiled extension

answer = Newton(example.callback(), example.derivative_callback(), 1.5).run()
assert answer.success
```

Passing `example.residual` itself also works, but as a Python callback: every
call goes through Python's call machinery and needs the GIL. The descriptor
returned by `example.callback()` points at the native C entry point instead.
`nogil` allows calls without the GIL, and the solver releases it around the
numerical work. The example functions have no state, so their contexts are null
and their function wrappers serve as owners; functions with parameters need a
native context and an owner that keeps it alive.

## How This Differs From Numba

The recommended Cython route is ahead of time: Cython translates the source to
C or C++, and a compiler builds an importable extension. Type annotations mark
the C arithmetic. Imported as ordinary Python, the same `.py` file does not
become native code, and the example's `cython.cimports` must be compiled before
it can be imported at all. Neither the `.py` nor the `.pyx` suffix decides
whether the GIL is needed.

Cython also has runtime compilation, such as `cython.inline` and
`@cython.compile`. These need a build toolchain, come with constraints on the
source, typing and Python state, and produce Python-callable functions. They do
not export CyRoot's callback ABI or guarantee GIL-free execution, so they are
not a drop-in equivalent of `numba.njit` for a fully native solve. CyRoot has
no runtime helper that automates this.

## Native Restrictions

To run without the GIL, the body must use native types and operations only.
Python objects, Python-level NumPy calls, dynamic closures and most Python
library calls need the GIL. C math functions can replace many scalar `math`
calls, and vector functions can write directly into the callback's output span
through typed pointers. Numerical derivatives can then sample the native
residual without returning to Python.

`exceptval(check=False)` is safe only when no exception can escape. A bridge
that can fail should check its own conditions and return a nonzero status,
which CyRoot reports as `NativeCallbackError`. Shared mutable native context must
support the concurrency you intend. Compiling a function does not make it
thread-safe, accurate or fast.

For the raw scalar, rectangular vector and Hessian interfaces, see the
[native core guide](native-core.md), [rectangular callbacks](native-rectangular-callbacks.md)
and [tensor callbacks](native-rectangular-tensors.md). The pure Python example
needs neither NumPy arrays nor NumPy C headers.

## Vector Functions And NumPy

The same [pure-Python example](../examples/pure_python_callback.md#vector-functions-in-the-same-py-module)
includes a compiled vector residual, analytic Jacobian and native numerical
Jacobian solve. It accepts two variables and writes three residuals directly
into solver-owned output storage. No ndarray is created per sample.

Compiling an existing NumPy function generally does not remove its Python
calls. For the native path, rewrite elementwise expressions as typed operations
or loops, use C math functions, and call native BLAS or LAPACK for matrix
operations. NumPy arrays remain useful for setup and for the results. Typed
memoryviews can be acquired once, outside the callback, and then used without
the GIL; with CyRoot's raw pointer interface there is no need to create a view
on every call. Other NumPy or SciPy routines need a native implementation of
their own, or a Python callback.
