# Native Vector Modified Halley

Install `cy-root`, Cython 3.3.0, setuptools, a C++17 compiler and matching Python
development headers. From this examples directory:

```sh
python -m pip install Cython==3.3.0 setuptools
python native_vector_modified_setup.py build_ext --inplace
python -c "from native_vector_modified import solve; print(solve(alpha=.25)[0]); print(solve(alpha=-.5, numerical=True)[0])"
```

The example solves \(x^2+xy+y=5\) and \(2x+y^2=6\) near \((1,2)\)
with native residual, Jacobian and Hessian callbacks. Each adapter retains its
parameter owner. Raw Jacobians use `output+n*input`; Hessians use
`output+n*(input1+n*input2)`. `numerical=True` uses native central derivative
plans with the same residual callback.

`VectorModifiedHalley` normalizes a finite real `alpha` once during guarded
construction. Its general path solves \(Js=F\), forms \(A=H[s,\cdot]\),
solves \((J-\alpha A)v=F\), and finally solves \(Jt=Av\) with the retained
Jacobian factors and pivots. The update is \(-s-t/2\). Matrix order matters.
The correction solve avoids cancellation from dividing \(v-s\) by small alpha.

Exact alpha values `0`, `.5` and `1` select the existing Chebyshev, Halley and
Super-Halley paths, including their workspace footprints. The solver's `method`
remains `"modified_halley"`. Other finite alpha values use two extra square
matrices and a second pivot vector. No automatic parameter choice or fallback
is supplied. Local convergence still depends on suitable roots and accurate
derivatives; some finite parameters can produce singular secondary systems.

Results contain copied root/residual tuples, `(Jacobian,Hessian)` derivative
tuples and `(residual,Jacobian,Hessian)` attempt counts. Retain a solver for
`step()`, `history`, `alpha` and diagnostics. `linear_stage` identifies
`"jacobian"`, `"secondary"` or `"correction"`; `linear_status` retains the
latest LAPACK status. Failed pretrial work preserves the last accepted point.

Fully native callbacks and kernels run without the GIL. The compiled example
needs no NumPy headers. The package retains NumPy and SciPy, and this native
engine supports square and rectangular residuals; this example uses the square
callback ABI.

The established square higher-order API accepts the same native callbacks:

```python
from native_vector_modified import solve_established

for method in ('halley', 'super_halley', 'chebyshev', 'tangent_hyperbolas'):
    result = solve_established(method)
    assert result.success
modified = solve_established('halley', alpha=.25, numerical=True)
assert modified.success
```

These public names now use matrix semantics, tuple snapshots and native plans.
See the [migration guide](../docs/vector-higher-order-migration.md) for the
removed options and the square versus rectangular linear-solve policies.
