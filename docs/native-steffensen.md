# Native Traub–Steffensen

`cyroot.traub_steffensen` now uses the shared native Newton state with
residual-dependent divided differences. The numerical loop, derivative samples
and square LU solves use native buffers and SciPy's Cython LAPACK provider.
Native callbacks allow numerical work without the GIL. Python callbacks and
input/result conversions use the GIL; NumPy and SciPy remain dependencies.

```python
from cyroot import traub_steffensen
from cyroot.native import TraubSteffensen

def residual(point):
    return (point[0]**2-2.,)

result = traub_steffensen(residual, (1.5,))
assert result.success
assert abs(result.root[0]-2.**.5) < 1e-9

solver = TraubSteffensen(residual, (1.5,), history=16)
first = solver.step()
result = solver.run()
assert result.success
assert first.iters == 1
assert solver.history[-1].root == result.root
```

The established signature is
`traub_steffensen(F, x0, F_x0=None, *, etol=1e-10, ertol=4*eps,
ptol=1e-12, prtol=4*eps, max_iter=200, history=0, history_every=1,
history_bytes=8*1024*1024, workspace_bytes=64*1024*1024)`.
The module export and `find_vector_root('traub_steffensen', ...)` share this
behavior. Options after `F_x0` are now keyword-only.

Retain `cyroot.native.TraubSteffensen(function, x0, f_x0=None, **options)` for
`step()`, `run()`, iteration and copied `.history`. It subclasses `VectorNewton`
and uses the same stopping, resource and failure rules; it has no analytic
Jacobian role or Jacobian cache. The native `traub_steffensen(function, x0,
**options)` convenience function runs one state. Shapes must be nonempty square
systems: point and residual both have length \(n\). An explicit
`RectangularCallback` is accepted when its input and output dimensions match.

## Residual-dependent differences

At an accepted point \(x\), take the signed coordinate step \(h_j=F_j(x)\).
For each axis, evaluate \(F\) at the floating-point point obtained by adding
\(h_j\) to coordinate \(j\). If its actual rounded displacement is \(a_j\),
the matrix column is

\[
D_{ij}(x)=\frac{F_i(x+a_je_j)-F_i(x)}{a_j}.
\]

The next point solves \(D(x)\delta=-F(x)\), then evaluates \(x+\delta\).
The steps are recomputed from the residual at every accepted iterate. They are
not the automatic small steps of `JacobianPlan`. In one dimension this is the
usual Steffensen update. In multiple dimensions, coordinate/residual pairing
makes the method depend on the ordering and scaling of the equations.

This implementation shares the native forward-difference stencil with
`JacobianPlan`. It divides by the actual displacement instead of the requested
residual component, correcting the old formula when rounding changes the step.
All nodes are validated before any sample. A zero residual component at a
nonroot, or an addition that rounds back to the original coordinate, reports
unsuccessful `breakdown`. An overflowing node or displacement reports
`nonfinite`. A zero component does not imply a zero Jacobian column; using an
inverse of a singular perturbation matrix would hide that missing information.
There is no automatic finite-difference fallback. For systems with such
components, `generalized_newton(F, None, x0)` supplies independently sized steps.
An initially converged full residual skips the unused differences entirely.

Required singular linear systems report `breakdown` under the existing strict
square LU policy. There is no pseudoinverse, rectangular extension, damping,
line search or global convergence guarantee. Only residual convergence reports
root success; a small step with a nonzero residual is unsuccessful stagnation.

## Results and counts

Python callbacks now receive immutable tuples, so replace NumPy-specific point
arithmetic with indexed sequence operations or explicit conversion inside the
callback. Inputs and outputs accept sized numeric sequences or native-format
double buffers. Results now use `NewtonMethodsReturnType`, with copied tuple
point/residual/Jacobian rows, and `f_calls=(residual_calls, jacobian_attempts)`.
This replaces the old `RootReturnType` and scalar call count.

A complete divided matrix costs \(n\) residual probes. Its base value is reused;
one uncached initial base evaluation adds one call, and cached `F_x0` adds zero.
`jacobian_attempts` counts each matrix attempt, including a preflight failure
that makes no sample calls. Trial residuals count as iterations; derivative
probes do not. After a terminal accepted trial, no unused matrix is generated
and `df_root` contains NaNs. These rules match [native Newton](native-vector.md).

`step()` and `.history` return `VectorNewtonStep`, including matrix-at-point,
counts, callback stage/code, linear status and rank (`None` for square LU).
A failed probe reports native callback stage `'jacobian'`; base/trial failures
report `'function'`. Original Python exceptions retain their identity. A failed
trial preserves the previous accepted point/residual/matrix. A failed matrix
rebuild preserves the newly accepted point/residual with a completely NaN
matrix. Terminal inspection remains passive after exception delivery.

Native workspace is the square numerical-Newton budget: \(10n+2n^2\) doubles
and \(n\) C-int pivots. The shared stencil's \(5n\)-double scratch is included;
there is no persistent step vector or derivative-plan allocation. Native
storage is allocated before iteration and reused. History is separately bounded;
Python boundary copies and backend-internal memory are excluded from the native
budget. Callback reservations span setup, samples and result/history export.

The private `_steffensen_jacobian` declaration in `derivatives.pxd` follows the
trusted disjoint `x[n]`, `base[n]`, `out[n*n]`, `work[5*n]` contract, with a square
native residual function/context and call/status pointers. The caller validates
positive dimensions/products and pins all buffers and callback lifetime. Failed
computations poison the whole output and restore the scratch point. No new
callback descriptor or extension module is needed. Compiled consumers must
rebuild for the draft solver's added private flag; callback, record and result
layouts are otherwise unchanged.

The legacy NumPy Traub–Steffensen loop is removed. [Barnes](native-barnes.md), [Robinson](native-robinson.md) and
[Wolfe–Bittner](native-wolfe-bittner.md) also use native states.
