# Native Scalar Finite Differences

Install `cy-root`, Cython 3.3.0, setuptools, a C++17 compiler, and matching Python
development headers. From this examples directory:

```sh
python native_scalar_derivative_setup.py build_ext --inplace
python -c "from native_scalar_derivative import solve; result, history = solve(); print(result)"
```

The example supplies a native residual for \(x^2-2=0\) and computes its first
derivative using `ScalarDerivativePlan()`. The native callback and derivative
arithmetic execute without the GIL. `Newton(function, None, x0)` also chooses a
default central plan; Python residuals acquire the GIL for each sample.

`scheme` accepts `"forward"`, `"backward"`, or `"central"`. The default relative
step adapts to the current point; an explicit signed finite nonzero `step`
chooses an absolute displacement, for example `solve(step=-1e-4)`. Coordinates
and their actual rounded offsets are validated before sampling. Collapsed or
nonfinite stencils terminate with a diagnostic instead of changing the stencil.

The immutable plan owns no callback or workspace and can serve independent
solves concurrently. Forward and backward plans make one additional residual
call per derivative; central plans make two. The first entry of result
`f_calls` includes every base, trial, and stencil residual attempt; the second
counts derivative attempts. Cached initial values skip their evaluations, and
stopping points skip unused derivatives. A stencil callback failure has
`callback_stage="derivative"`, preserving the original Python exception or native
status along with the last accepted point and residual.
