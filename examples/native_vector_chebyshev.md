# Native Vector Chebyshev

Install `cy-root`, Cython 3.3.0, setuptools, a C++17 compiler and matching Python
development headers. From this examples directory:

```sh
python -m pip install Cython==3.3.0 setuptools
python native_vector_chebyshev_setup.py build_ext --inplace
python -c "from native_vector_chebyshev import solve; print(solve()[0]); print(solve(numerical=True)[0])"
```

The example solves \(x^2+xy+y=5\) and \(2x+y^2=6\) near \((1,2)\).
It provides native residual, Jacobian and Hessian callbacks. Jacobians use raw
column-major indexing `output+n*input`; Hessians use
`output+n*(input1+n*input2)`. Each adapter retains its parameter owner.

Chebyshev factors the Jacobian once per iteration, solves \(Js=F\), contracts
\(q=H[s,s]\), solves \(Jt=q\) using the retained factorization, and steps by
\(-s-t/2\). This is a coupled vector method with local cubic convergence at
suitable simple roots. Derivative accuracy and floating-point limits affect
that behavior.

`numerical=True` uses default native `JacobianPlan` and `HessianPlan` instances.
Plans and analytic callbacks can also be mixed. The result's `df_root` holds
`(Jacobian,Hessian)` copied nested tuples; `f_calls` holds residual, Jacobian
and Hessian attempts. Numerical samples add to the residual count. Final
stopping points skip unused derivative evaluations.

Use a retained `VectorChebyshev` instance for `step()`, `history` and terminal
diagnostics. Snapshots are immutable copies. History and workspace have
separate byte limits. Fully native callbacks and numerical kernels run without
the GIL; Python callbacks reacquire it per evaluation. This example needs no
NumPy headers. The package retains its NumPy and SciPy dependencies, and the
native solver currently supports square systems.
