# Native Vector Newton Callbacks

Install `cy-root`, Cython 3.3.0, setuptools, a C++17 compiler, and matching Python
development headers. From this examples directory:

```sh
python -m pip install Cython==3.3.0 setuptools
python native_vector_setup.py build_ext --inplace
python -c "from native_vector import solve; result, history = solve(); print(result)"
```

The example solves the coupled equations
\(x^2+y=3\) and \(2x+y^2=6\), starting near the root \((1,2)\).
The analytic Jacobian is non-symmetric. Native callbacks receive a borrowed
read-only point, its dimension, an output pointer, and a context pointer. The
residual fills `n` values; the Jacobian fills `n*n` values in **column-major**
order, at `out[row + column*n]`. Each callback must fill every output entry and
return zero on success. Return a nonzero status on failure; do not let Python or
C++ exceptions cross the callback interface.

`make_vector_callback(..., jacobian=True)` marks the Jacobian adapter. Both
adapters retain the `CoupledSystem` instance, so the context remains valid after
local callback or owner variables go out of scope. Separate adapters sharing
mutable context require synchronization if used concurrently. The full native
run releases the GIL; Python callbacks remain supported through a bridge that
reacquires it and supplies an immutable point tuple.

`VectorNewton` currently accepts nonempty square real systems with analytic
Jacobians. SciPy's native LAPACK interface solves the dense linear system;
singular matrices produce an unsuccessful `breakdown` with `linear_status`.
This example does not use NumPy arrays or NumPy C headers, although the package
still depends on NumPy and SciPy. Results and recorded steps are copied tuples.
The final Jacobian contains NaNs when the final point did not need another
Jacobian evaluation; `f_calls` counts residual and Jacobian calls separately.
