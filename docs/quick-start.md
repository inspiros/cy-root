# Quick Start

CyRoot solves \(f(x)=0\) and \(F(x)=0\) with the same progression: choose a
method, provide the derivatives you have, and check the stopping reason.
Python callbacks are enough to begin. Native callbacks are an optional next step
when Python call overhead matters.

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

A bracket is a good starting point for a continuous scalar function with
opposite endpoint signs. Newton is useful when a starting estimate and derivative
are available. Pass `None` to approximate the derivative numerically.

```python
from cyroot import bisect, newton

f = lambda x: x*x - 2.
bracketed = bisect(f, 1., 2.)
analytic = newton(f, lambda x: 2*x, 1.5)
numerical = newton(f, None, 1.5)
assert bracketed.success and analytic.success and numerical.success
assert abs(analytic.root - 2.**.5) < 1e-10
```

Controls are keyword-only: `newton(f, None, 1.5, etol=1e-12, max_iter=50)`.
Cached initial function and derivative values must match the starting point.

## Solve A Nonlinear System

Vector callbacks receive a tuple. Return a sequence of residuals, and if
supplying a Jacobian, return rows indexed by residual and columns by input.
Tall systems have more residuals than unknowns; wide systems have fewer.

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

`history=0` disables recording. Positive capacities bound the number of copied
snapshots; `history_truncated` reports a full buffer. `RootSnapshot` normalizes
single-root scalar and vector records and keeps the original diagnostics.
Read [Stopping And Results](stopping.md) before interpreting `success`: a narrow
scalar bracket and a small nonlinear residual are distinct outcomes.

## Take The Next Step

- [Choose A Method](choosing-methods.md) by shape, derivatives, and storage.
- [Numerical Derivatives](numerical-derivatives.md) compares fixed differences,
  Richardson refinement, and complex-step.
- [Compile Callbacks](compiling-callbacks.md) explains the Cython route to a
  numerical loop without the GIL.
- [Native Core](native-core.md) documents ownership, storage, and callback errors.
