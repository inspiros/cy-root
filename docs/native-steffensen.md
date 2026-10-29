# Native Traub–Steffensen

`cyroot.traub_steffensen` runs the shared native Newton state with divided
differences whose steps come from the residual. The numerical loop, derivative
samples and square LU solves use native buffers and SciPy's Cython LAPACK
interface. With native callbacks the numerical work runs without the GIL;
Python callbacks and input and result conversions hold it.

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

The public signature is
`traub_steffensen(F, x0, F_x0=None, *, etol=1e-10, ertol=4*eps,
ptol=1e-12, prtol=4*eps, max_iter=200, history=0, history_every=1,
history_bytes=8*1024*1024, workspace_bytes=64*1024*1024)`, also available as
`find_vector_root('traub_steffensen', ...)`. Options after `F_x0` are
keyword-only.

`cyroot.native.TraubSteffensen(function, x0, f_x0=None, **options)` provides
`step()`, `run()`, iteration and copied `.history`, and
`cyroot.native.traub_steffensen(function, x0, **options)` runs one solver to
completion. It subclasses `VectorNewton` and shares its stopping, resource and
failure rules, without a Jacobian provider or Jacobian cache. Systems must be
square and nonempty, with point and residual both of length \(n\). A
`RectangularCallback` is accepted when its input and output dimensions are
equal.

## Residual-Based Differences

At an accepted point \(x\), take the signed coordinate step \(h_j=F_j(x)\).
For each axis, evaluate \(F\) at the floating-point point obtained by adding
\(h_j\) to coordinate \(j\). If its actual rounded displacement is \(a_j\),
the matrix column is

\[
D_{ij}(x)=\frac{F_i(x+a_je_j)-F_i(x)}{a_j}.
\]

The next point solves \(D(x)\delta=-F(x)\) and evaluates \(x+\delta\). The
steps are recomputed from the residual at every accepted iterate; they are not
the small automatic steps of `JacobianPlan`. In one dimension this is the usual
Steffensen update. In several dimensions, pairing coordinate \(j\) with
residual component \(j\) makes the method depend on the order and scaling of
the equations.

The differences use the same native forward-difference stencil as
`JacobianPlan`, dividing by the actual displacement rather than the requested
residual component, which the 1.x formula got wrong when rounding changed the
step. Every node is validated before any sample. A zero residual component away
from a root, or a step that rounds back to the original coordinate, reports an
unsuccessful `breakdown`; an overflowing node or displacement reports
`nonfinite`. A zero component says nothing about the Jacobian column, and
inverting a singular perturbation matrix would hide that missing information,
so there is no automatic fallback. For such systems, `generalized_newton(F,
None, x0)` uses independently sized steps. If the starting point is already a
root, no differences are computed.

A singular linear system reports `breakdown`, as for the square LU solve of
vector Newton. There is no pseudoinverse, rectangular variant, damping or line
search, and no global convergence guarantee. Only residual convergence counts as
a root; a small step with a nonzero residual is unsuccessful stagnation.

## Results And Counts

Python callbacks receive immutable tuples, so NumPy arithmetic on the point
needs indexed operations or an explicit conversion inside the callback. Inputs
and outputs can be sized numeric sequences or buffers. Results are
`NewtonMethodsReturnType` with copied tuple point, residual and Jacobian rows,
and `f_calls=(residual_calls, jacobian_attempts)`.

A complete divided-difference matrix costs \(n\) residual probes and reuses the
residual at the current point. An uncached initial residual adds one call; a
cached `F_x0` adds none. `jacobian_attempts` counts every matrix attempt,
including one that fails validation before sampling. Trial residuals count as
iterations; probes do not. When an accepted trial ends the solve, no matrix is
built for it and `df_root` contains NaNs. These rules match
[vector Newton](native-vector.md).

`step()` and `.history` return `VectorNewtonStep` records with the matrix at
the point, the counts, any callback stage and code, the linear status and the
rank (`None` for square LU). A failed probe reports the native callback stage
`'jacobian'`, a failed base or trial residual `'function'`. Python exceptions
propagate as the original instance. A failed trial keeps the previous accepted
point, residual and matrix; a failed matrix rebuild keeps the newly accepted
point and residual with an all-NaN matrix. Once an error has been raised,
further calls only return the solver's state.

The native workspace is that of numerical square Newton: \(10n+2n^2\) doubles
and \(n\) C-int pivots, including the stencil's \(5n\)-double scratch, with no
persistent step vector or derivative plan. It is allocated before the iteration
and reused. History is bounded separately. Python boundary copies and memory
used inside LAPACK are not counted. Callback reservations cover setup, samples
and result and history export.

The private `_steffensen_jacobian` declaration in `derivatives.pxd` takes
non-overlapping `x[n]`, `base[n]`, `out[n*n]` and `work[5*n]` spans, a square
native residual function and context, and call-count and status pointers. The
caller must validate positive dimensions and their products and keep every
buffer and callback alive. On failure the whole output is filled with NaN and
the scratch point is restored.
