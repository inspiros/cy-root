# Quick Start

CyRoot solves \(f(x)=0\) and \(F(x)=0\) the same way: choose a method, supply
the derivatives you have, and check why the solver stopped. Python callbacks are
enough to start. Native callbacks are an optional next step when Python call
overhead matters.

## Install

For a published 2.x release:

```sh
python -m pip install 'cy-root>=2,<3'
```

To build from a source checkout instead:

```sh
git clone https://github.com/inspiros/cy-root.git
cd cy-root
python -m pip install -e .
```

The package supports CPython 3.11–3.14 and requires NumPy and SciPy. A source
build also needs Cython 3.3.0, a C++17 compiler, and Python development headers.
See [Building And Testing](building.md) for the full environment setup.

## Solve A Scalar Equation

A bracket is a good starting point for a continuous scalar function whose values
at the two endpoints have opposite signs. Newton suits a good starting estimate
with a derivative; pass `None` to approximate the derivative numerically.

```python
from cyroot import bisect, newton

f = lambda x: x*x - 2.
bracketed = bisect(f, 1., 2.)
analytic = newton(f, lambda x: 2*x, 1.5)
numerical = newton(f, None, 1.5)
assert bracketed.success and analytic.success and numerical.success
assert abs(analytic.root - 2.**.5) < 1e-10
```

Tolerances and other controls are keyword-only:
`newton(f, None, 1.5, etol=1e-12, max_iter=50)`. Cached values such as `f_x0`
and `df_x0` are optional and must be the values at the starting point.

## Solve A Nonlinear System

Vector callbacks receive a tuple and return a sequence of residuals. A Jacobian
callback returns one row per residual and one column per unknown. Tall systems
have more residuals than unknowns; wide systems have fewer.

```python
from cyroot import generalized_newton, vector_lbfgs

f = lambda x: (x[0] - 1., x[1] - 2., x[0] + x[1] - 3.)
answer = generalized_newton(f, None, (0., 0.))
assert answer.success
assert max(abs(a-b) for a, b in zip(answer.root, (1., 2.))) < 1e-9

# Optimizer conveniences use the same function/Jacobian/point order.
optimized = vector_lbfgs(f, None, (0., 0.),
                         input_dimension=2, output_dimension=3)
assert optimized.success
```

Rectangular methods compute least-squares steps and still require the original
residual to meet tolerance. Some quasi-Newton methods require a square system;
see [Choose A Method](choosing-methods.md). For large matrix-free systems, see
[Newton–Krylov](native-newton-krylov.md).

## Inspect The Iterations

```python
from cyroot import as_root_snapshot
from cyroot.native import DampedNewton

solver = DampedNewton(lambda x: x*x - 2., None, .1, history=32)
first = as_root_snapshot(solver.step())
final = as_root_snapshot(solver.run())
assert final.residual_converged
assert solver.history[-1].root == final.root
assert first.iters == 1
```

`history=0`, the default, records nothing. A positive `history` is the maximum
number of copied snapshots; once it is full, later steps are not recorded and
`history_truncated` is set. `as_root_snapshot` gives scalar and vector results
the same fields and keeps the original record in `diagnostics`.
Read [Stopping And Results](stopping.md) before interpreting `success`: a narrow
scalar bracket and a small nonlinear residual are different outcomes.

## Take The Next Step

- [Choose A Method](choosing-methods.md) by system shape, available derivatives
  and storage.
- [Numerical Derivatives](numerical-derivatives.md) compares fixed differences,
  Richardson refinement and complex-step derivatives.
- [Compile Python Callbacks](compiling-callbacks.md) shows how to run the
  numerical loop without the GIL.
- [Native Core](native-core.md) documents ownership, storage, and callback errors.
