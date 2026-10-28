# Native higher derivatives and a Householder increment

Install cy-root, Cython 3.3.0, setuptools, matching Python development headers
and a C++17 compiler. From this directory:

```sh
python native_higher_derivative_setup.py build_ext --inplace
python -c "from native_higher_derivative import estimate_exponential, quadratic_householder_step; print(estimate_exponential()); print(quadratic_householder_step(12))"
```

`estimate_exponential` samples a compiled exponential callback with a
runtime-order finite-difference plan. The third derivative at 0.4 approximates
`exp(0.4)`. Its native samples and arithmetic execute without the GIL. The
Python convenience boundary allocates checked scratch and returns an immutable
estimate with the residual-attempt count.

`quadratic_householder_step` supplies analytic derivatives of `x²-1` to the
native reciprocal-series kernel. Its default order twelve already exceeds
the legacy explicit formula table. No symbolic algebra is generated or evaluated.
The returned tuple contains native status, increment and next point; it is
one step, not a complete root solve. Its configurable example workspace budget
is checked before vector allocation. The C++ kernel itself allocates nothing.

Both paths use installed `.pxd` declarations and native headers without NumPy
C headers. Larger orders remain subject to conditioning, derivative accuracy,
finite input range and configured resources. See the
[manual](../docs/native-higher-derivative.md) for the recurrence, status and
ownership contracts. Package dependencies still include NumPy/SciPy/SymPy.
