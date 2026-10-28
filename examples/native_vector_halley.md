# Native matrix Halley and Super-Halley

Install `cy-root`, Cython 3.3.0, setuptools, a C++17 compiler and matching Python
development headers. From this examples directory:

```sh
python -m pip install Cython==3.3.0 setuptools
python native_vector_halley_setup.py build_ext --inplace
python -c "from native_vector_halley import solve; print(solve()[0]); print(solve(method='super_halley')[0])"
```

The example solves \(x^2+xy+y=5\) and \(2x+y^2=6\) near \((1,2)\).
It supplies native residual, Jacobian and Hessian callbacks. Each adapter
retains its parameter owner. Raw Jacobians use `output+n*input`; Hessians use
`output+n*(input1+n*input2)`. Analytic Hessians supply every ordered entry.

Both fixed solver classes first solve \(Js=F\) and form the directional
Hessian matrix \(A=H[s,\cdot]\). `VectorHalley` solves
\((J-A/2)v=F\) and steps by \(-v\). `VectorSuperHalley` solves
\((J-A)v=F\) and steps by \(-(s+v)/2\). Matrix order matters for coupled
systems. Local cubic convergence requires suitable simple roots and accurate
derivatives; finite estimates alone do not certify that accuracy.

`solve(numerical=True)` selects native central Jacobian/Hessian plans. Analytic
and numerical providers can also be mixed. `df_root` holds the copied
`(Jacobian,Hessian)` pair; `f_calls` counts residual, Jacobian and Hessian
attempts. Numerical samples add to the residual count.

Retain a solver instance for `step()`, `history`, `method` and `linear_stage`.
History records are copied immutable `VectorHigherOrderStep` tuples, also
available by their existing `VectorChebyshevStep` name. `linear_stage` identifies
the Jacobian or secondary linear system, while `linear_status` retains LAPACK's
status. A singular secondary system terminates with `breakdown` and no trial.

The matrix methods share the Chebyshev state engine and add one private square
matrix to its workspace. Fully native callbacks run without the GIL; Python
callbacks reacquire it per evaluation. No NumPy headers are needed by this
example. The package retains NumPy and SciPy, and these native methods support
square residuals. The older componentwise Halley APIs remain available.
