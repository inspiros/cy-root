# Native finite-difference Jacobians

Install `cy-root`, Cython 3.3.0, setuptools, a C++17 compiler, and matching Python
development headers. From this examples directory:

```sh
python -m pip install Cython==3.3.0 setuptools
python native_jacobian_setup.py build_ext --inplace
python -c "from native_jacobian import solve; result, history = solve(); print(result)"
```

The example supplies only a native residual for the coupled equations
\(x^2+y=3\) and \(2x+y^2=6\). `JacobianPlan(2)` supplies a central difference
Jacobian, and `VectorNewton` converges near \((1,2)\). Passing `None` as the
solver's Jacobian also creates a default central plan.

A plan is immutable and reusable. `scheme` accepts `"forward"`, `"backward"`,
or `"central"`. `step=None` chooses steps from each coordinate's current
magnitude. A scalar or length-n sequence supplies signed, nonzero absolute
steps; for example `solve(step=(1e-5, -2e-5))`. Plans preflight every stencil
coordinate before sampling. Collapsed or nonfinite displacements terminate the
solve with a diagnostic rather than adjusting the stencil silently.

Forward/backward plans make n additional residual calls per Jacobian; central
plans make 2n. All use the already available base residual. Result `f_calls`
counts every base, trial, and stencil residual evaluation in its first entry,
and Jacobian attempts in its second entry. Accepted initial roots and final
stopping points skip unused Jacobians. A stencil callback failure is reported
at `callback_stage="jacobian"`; the original Python exception or native status
is preserved, and the retained point/residual are still available.

The native plan reuses private solver scratch and makes no Python calls when
the residual is native. Python residuals use the same plan while acquiring the
GIL for each callback. The package still depends on NumPy and SciPy, and this
checkpoint supports square systems rather than arbitrary derivative tensors.
