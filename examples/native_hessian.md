# Native finite-difference Hessians

Install `cy-root`, Cython 3.3.0, setuptools, a C++17 compiler and matching Python
development headers. From this examples directory:

```sh
python -m pip install Cython==3.3.0 setuptools
python native_hessian_setup.py build_ext --inplace
python -c "from native_hessian import estimate; print(estimate())"
```

The native residual has two quadratic components. Its copied Hessian tensor is
indexed `[residual][input][input]`, and should be close to
`(((2, 3), (3, 4)), ((10, -7), (-7, 22)))`. Input-axis mirrors are copied exactly.
The callback adapter retains its context owner throughout evaluation.

`HessianPlan(2)` uses central differences with automatic steps. `scheme` also
accepts `"forward"` and `"backward"`; a signed scalar or length-n step sequence
supplies explicit absolute steps. For example,
`estimate(step=(2**-8, -(2**-9)))`. Actual rounded stencil coordinates determine
the interpolation weights. Very small steps can collapse and return a
`"breakdown"` status; finite estimates can still have numerical error.

The immutable result contains `hessian`, `f_calls`, `status` and `callback_code`.
Central evaluation makes `2*n*n` stencil calls; one-sided evaluation makes
`n*(n+3)//2`. The base adds one call unless passed as `f_x` to `evaluate`.
Axis samples are cached before mixed corners are evaluated. Failed evaluation
poisons the whole tensor; Python exceptions and native error codes are retained.

The plan is immutable and reusable with independent callbacks and scratch.
`memory_bytes` limits its steps plus the native storage needed for one
evaluation; copied Python results and simultaneous evaluations use additional
memory. Fully native residual evaluation releases the GIL. Python residuals
use the same plan and reacquire the GIL for each callback.

This example needs no NumPy headers. The package still depends on NumPy and
SciPy. This stage provides standalone Hessians for square residuals; solver
integration and rectangular residuals are separate work.
