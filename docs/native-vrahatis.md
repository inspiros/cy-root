# Native Vrahatis Generalized Bisection

`cyroot.vrahatis` now runs a native state machine. The input is either a matrix
of sign-polygon vertices or one point from which a bounded heuristic attempts to
construct a polygon. The established function name and main argument order are
retained. Python callbacks receive copied point tuples and return a residual
vector. Lists, tuples and compatible NumPy buffers are accepted at setup; result
and history data are copied into immutable tuples.

```python
from itertools import product
from cyroot import vrahatis
from cyroot.native import Vrahatis

vertices = tuple(product((0., 2.), repeat=2))
def residual(x):
    return (x[0]-.7, x[1]-.7)

answer = vrahatis(residual, vertices)
assert answer.success

solver = Vrahatis(residual, vertices, history=64)  # Keep the terminal sweep too.
first = solver.step()
answer = solver.run()
assert answer.success
assert solver.history[-1].root == answer.root
```

## Sign And Stopping Contracts

For a residual with `m` components and points with `n` coordinates, an ordinary
starting polygon must have exactly `2**m` vertices, one for every strict output
sign pattern. Residual components at those vertices must be finite and nonzero.
A supplied vertex already meeting the residual tolerance can return before the
sign-polygon condition is required. All input and cached residual rows must have
matching dimensions. Cached `F_x0s` values avoid new callback evaluations.

Canonical vertex order is the binary order of the signs, negative before positive.
Each sweep visits output-sign cells that differ in one bit. It samples the
midpoint of their current representatives, then replaces a sign-compatible
representative. A zero residual component is compatible with either sign. An
unchanged candidate causes stagnation; a failed callback or nonfinite sample
preserves the last accepted finite state.

`precision` is the largest coordinate span across the polygon, not the residual
spread or Euclidean diameter. Residual error is the infinity norm. Relative
residual tolerance uses the initial residual scale. The iteration budget counts
sweeps; a sweep can sample several edges, and stops early at a terminal condition.
Function counts include setup and polygon-search samples, but not supplied
cached values. For a constructed polygon, its retained vertex residuals establish
the iteration residual scale. A residual-converged sample found during discovery
can terminate immediately without completing the strict polygon.

For one output component, a sign-changing segment gives the usual scalar bracket
certificate under continuity; a small coordinate width can terminate successfully.
For multiple output components, complete vertex signs alone **do not certify a
common root exists**. Small width is therefore stagnation, not root success.
Only residual tolerance gives successful multidimensional termination. This is a
sign-based root-search method; the implementation does not infer a face/degree
certificate from its vertices.

Rectangular `n` and `m` are supported by the same sign-cell recurrence. A polygon
still needs `2**m` output patterns. Extra input coordinates remain ordinary
coordinates in its geometry; this is not a uniqueness or existence theorem for
rectangular systems.

## Python And Compiled Callbacks

Pass an ordinary Python callable, a residual `VectorCallback` for square systems,
or a residual `RectangularCallback` for different input/output lengths. Native
producers use `make_vector_callback` or `make_rectangular_callback` in the
existing declaration modules. There is no new callback ABI.

A native residual writes all output elements to the supplied pointer and returns
zero on success. Nonzero status raises `NativeCallbackError`. The producer must
pin its context using the descriptor's keepalive, avoid escaping exceptions and
synchronize any shared mutable native state. Reentrant or concurrent use of one
adapter is rejected. Original Python exceptions retain their identity.

Setup conversions and copied exports use the GIL. With native callbacks, vertex
initialization, polygon sampling and entire numerical `run()` execute outside the
GIL without NumPy calls or per-sample array/view creation. Python callbacks acquire
the GIL only through their bridge. See [compiling callbacks](compiling-callbacks.md)
for typed Python-syntax and Cython examples.

## Single-Point Polygon Discovery

```python
from cyroot import vrahatis
from cyroot.vector_bracketing import compute_admissible_n_polygon

# A cube from (-1,-1) with side steps (2,2) already covers all four signs.
points, signs = compute_admissible_n_polygon(
    lambda x: (x[0]-.2, x[1]-.3), (-1.,-1.), h=(2.,2.))
answer = vrahatis(lambda x: (x[0]-.2, x[1]-.3), points)
assert answer.success
```

`h` is a scalar or one finite step per input coordinate. The default uses `max(1, norm_inf(x))` for every coordinate, including at the
origin. The search samples one
axis-aligned input cube, then refines actual component sign crossings on its
edges to discover missing strict patterns. A complete polygon keeps one point
per pattern together with its actual cached residual. It does not manufacture
signs, count zero components as extra patterns, or repeatedly resample completed
vertices before iteration.

The old helper selected true input-cube edges only in two dimensions; the native
helper selects edges by a single changed input bit in every supported dimension.
Same-sign component endpoints are skipped, rather than passed to scalar bisection.
The search is bounded and heuristic: failure to find a complete polygon is not
proof that no root or admissible polygon exists. Supply explicit vertices or
change the initial point/steps when construction fails.

`polygon_eps` controls the internal crossing search. The compatibility helper
calls it `eps`. Search budgets `max_evaluations`, `max_visits` and `max_bisections`
are separate from solver `max_iter`; resource limits and finite steps are checked
before unchecked native loops. Exponential `2**n` input corners and `2**m` output
patterns make large dimensions expensive even with native loops.

The standalone polygon helper requires a complete strict polygon even when its
search samples a root; use the solver to return such a root directly.

The standalone helpers `get_M`, `_get_1_simplexes` and `sorted_by_vertices` remain
available from `cyroot.vector_bracketing`. Their NumPy arrays are compatibility
boundary outputs. The solver and native polygon builder do not call those array
helpers or NumPy numerical operations internally.

## Inspection And Resources

`Vrahatis` exposes `step()`, `run()`, `result`, `done`, iteration, `history` and
`history_truncated`. A step is one edge sweep. History stores copied root,
residual, polygon, residual polygon, counters and metrics. Use `history=0` to
disable it; `history_every`, `history_bytes` and capacity bound retained storage.
A full history does not stop the numerical solve. Earlier snapshots remain
valid after resumption or solver destruction.

`workspace_bytes` bounds native logical workspace, and `history_bytes` bounds
native history storage. Python conversion objects, tuple exports, allocator
metadata and callback-owned allocations are additional memory. These are not
process peak-memory guarantees. Inputs are copied; caller arrays are not mutated.

## Callback Inputs And Scope

NumPy callbacks such as `lambda x: x-target` must use explicit elementwise Python
expressions or convert their tuple input when needed; such conversion still uses
Python. Return types now contain tuples, and counters/iteration paths may differ
from the legacy implementation. Rebuild compiled consumers against the current
2.0 declarations. Private `vrahatis_kernel` is removed.

Vrahatis completes the native migration of currently supported vector solvers.
The unverified experimental Mishchenko and Ahmadi methods were removed. This
coverage does not complete cross-platform or release qualification. A future
Eiger–Sikorski–Stenger solver needs separate degree and simplex-search machinery;
it is not implemented by this sign-polygon method.
