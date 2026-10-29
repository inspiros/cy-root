# Native Least Squares

Install cy-root, Cython 3.3.0, setuptools and a C++17 compiler with matching
Python development headers. Build against the installed package:

```sh
python native_least_squares_setup.py build_ext --inplace
python -c "from native_least_squares import solve; print(solve()); print(solve(True))"
```

The tall case minimizes the residual of
`A = ((1,0), (0,1), (1,1))`, `b = (1,2,4)`. Its solution is approximately
`(4/3, 7/3)`, with rank two and a nonzero residual. The wide case uses
`A = ((1,0,1), (0,1,1))`, `b = (1,2)`. Its minimum-norm solution is
approximately `(0,1,1)`, also with rank two.

Both cases create an immutable `LeastSquaresPlan`, allocate all input, output,
scratch and pivot storage once and call `compute` twice in a `nogil` block.
The same scratch is reused; the plan resets its pivot and copied input data
on every call. Returned `status == 0`, `info == 0` and `gil_held == False`
are expected. Tuple and dictionary construction occurs after reacquiring
the GIL. Parallel calls may share a plan with separate workspaces.

The convenience Python API allocates private workspaces on each call:

```python
from cyroot.native import LeastSquaresPlan

result = LeastSquaresPlan(2, 3).solve(((1., 0., 1.), (0., 1., 1.)), (1., 2.))
assert result.status == 'ok'
assert result.rank == 2
```

The example needs no NumPy headers; cy-root still depends on NumPy and SciPy.
Read the [manual](../docs/native-least-squares.md) for the rank threshold,
transactional outputs, trusted raw pointer contract and memory accounting.
A successful linear solve is not a nonlinear root-convergence result.
