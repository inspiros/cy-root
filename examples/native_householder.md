# Native Householder Callbacks

Install `cy-root`, Cython 3.3.0, setuptools, a C++17 compiler, and matching Python
development headers. From this examples directory:

```sh
python -m pip install Cython==3.3.0 setuptools
python native_householder_setup.py build_ext --inplace
python -c "from native_householder import solve; print(solve(order=3)[0]); print(solve(order=12)[0])"
```

The example solves \(x^2=2\) with native residual and derivative callbacks.
It supports a runtime derivative count: first derivative \(2x\), second
\(2\), and zero derivatives thereafter. The implementation uses a reciprocal
Taylor recurrence rather than a table of formulas. Sharing the zero-derivative
adapter across several roles is supported. Every adapter retains the context
owner so its native parameter pointer remains valid throughout the solve.

`Householder` accepts a list/tuple of analytic callbacks, numerical plans or
`None` entries; `derivatives=None, order=d` supplies default numerical plans.
Generic numerical derivatives can be inaccurate at high orders. Resource
limits bound node counts, workspace and history; analytic derivatives have no
fixed formula-table order cap. The derivative count d ordinarily gives local
convergence order d+1 at a suitable simple root, subject to derivative accuracy
and floating-point limits.

`run()` returns a copied result whose `df_root` is a derivative tuple and
`f_calls` counts the residual followed by each derivative role. Terminal points
skip unused derivative work. `step()` and `history` expose immutable snapshots
with callback stage and derivative order on failures. A fully native callback
run releases the GIL; Python callback bridges reacquire it per evaluation.
No NumPy arrays or NumPy C headers are needed by this example.
