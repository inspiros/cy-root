# Native Broyden

Install cy-root, Cython 3.3.0, setuptools and a C++17 compiler with matching
Python development headers. Build against the installed package:

```sh
python native_broyden_setup.py build_ext --inplace
python -c "from native_broyden import solve; print(solve('good')[0]); print(solve('bad')[0])"
```

The example solves \(x^2+y=2\), \(x+2y^2=3\) near \((1,1)\).
`BroydenGood` starts with an explicit Jacobian approximation. `BroydenBad`
starts with its explicitly supplied inverse; the solver does not invert a
Jacobian guess automatically. A pinned Cython owner carries the residual
parameters. Native callbacks and solve/update loops execute outside the GIL;
setup and tuple result/history conversion use it. No NumPy headers are needed
by this example; the package still depends on NumPy and SciPy.

Each copied result includes the retained secant approximation, its kind, and
`approximation_iteration`, the last accepted iteration that updated it. A
terminal root or budget stops before an unused update, so this counter can
lag `iters`. `phase` identifies linear, trial or update work. Update failure
retains the accepted point/residual and previous complete approximation.

Read the [native Broyden manual](../docs/native-broyden.md) for the update
equations, square-system policy, resource limits and failure semantics.

The same native residual can initialize the approximation without Python sample
calls. `solve_initialized(method, scheme, cached)` uses a `JacobianPlan`; the
bad variant calls the explicit inverse factory:

```python
from native_broyden import solve_initialized

for method in ('good', 'bad'):
    result, history, initial = solve_initialized(method, 'central', cached=True)
    assert initial.initialization_calls == 1
    assert initial.approximation_valid
    assert result.success
```

With two inputs, forward/backward initialization uses two residual samples and
central uses four, in addition to an uncached base evaluation. All subsequent
iterations reuse the secant approximation. Recompile consumers against current
declarations; result tuples append `approximation_valid` and `initialization_calls`.
