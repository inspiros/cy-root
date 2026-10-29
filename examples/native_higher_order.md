# Native Halley Family Callbacks

Install `cy-root`, Cython 3.3.0, setuptools, a C++17 compiler, and matching Python
development headers. From this examples directory:

```sh
python -m pip install Cython==3.3.0 setuptools
python native_higher_order_setup.py build_ext --inplace
python -c "from native_higher_order import solve; print(solve('halley')[0]); print(solve('chebyshev')[0])"
```

The example solves \(x^2=2\) using analytic residual, first derivative and
second derivative callbacks. Each native callback writes `out[0]` and returns
zero for success. `make_scalar_callback` retains the shared `Quadratic` owner,
keeping its native parameter context valid while any adapter/solver uses it.
Nonzero callback statuses become `NativeCallbackError` at the Python boundary;
Python or C++ exceptions must not cross the callback ABI.

The available `method` names are `'halley'`, `'chebyshev'`, `'super_halley'`,
`'modified_halley'` and `'tangent_hyperbolas'`. Select the modified family's
parameter with `solve('modified_halley', alpha=1.)`, for example. Scalar
tangent hyperbolas shares Halley's implementation and has no formula selector.

Pass `derivatives='numerical'` to `solve` to approximate both derivatives from
the native residual, or `'mixed'` to keep the analytic first derivative and
approximate the second. Both plans reuse the current residual but sample
independently. They execute without NumPy operations or the GIL. Numerical
derivative errors can limit the methods' analytic cubic convergence rate.

All methods support analytic/numerical and Python/native callback mixtures,
release the GIL for a complete native run,
and expose immutable copied steps/history. `f_calls` is the triple
`(residual, first_derivative, second_derivative)` for this solve. Each invoked
plan counts as one derivative attempt, with its samples also added to the
residual counter. Accepted
starting roots skip unused derivatives; terminal trials skip derivatives at
the returned point, where result `df_root` contains two NaNs. A derivative
failure keeps the accepted point/residual and identifies the failing stage.

Halley, Super-Halley and modified Halley can stop with `breakdown` when their
correction denominator vanishes. All methods return explicit failures for
nonfinite arithmetic. These methods add no damping, line search or fallback
Newton steps. The example
needs no NumPy arrays or C headers, although package dependencies remain.
