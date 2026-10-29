# Native Vrahatis Generalized Bisection

`cyroot.vrahatis` runs a native state machine. Its input is either a matrix of
sign-polygon vertices or a single point, from which a bounded heuristic tries to
build a polygon. Python callbacks receive a copied point tuple and return a
residual vector. Setup accepts lists, tuples and compatible NumPy buffers;
results and history are copied into immutable tuples.

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

For a residual with `m` components and points with `n` coordinates, a starting
polygon has exactly `2**m` vertices, one for each strict sign pattern of the
residual. Every residual component at those vertices must be finite and
nonzero, except that a supplied vertex that already meets the residual
tolerance is returned before the sign condition is checked. All input and
cached residual rows must have matching dimensions. Cached `F_x0s` values avoid
new callback evaluations.

Vertices are ordered by the binary order of their signs, negative before
positive. Each sweep visits pairs of sign cells that differ in one component,
samples the midpoint of their current vertices, and replaces the vertex whose
sign pattern the midpoint matches. A zero residual component matches either
sign. A candidate that equals its vertex causes stagnation; a failed callback
or nonfinite sample keeps the last accepted finite state.

`precision` is the largest coordinate span of the polygon, not the residual
spread or the Euclidean diameter. The residual error is the infinity norm, and
the relative residual tolerance uses the initial residual scale. The iteration
budget counts sweeps; one sweep can sample several edges and stops early at a
terminal condition. Function counts include setup and polygon-search samples,
but not supplied cached values. For a constructed polygon, the residuals of its
vertices set the residual scale. A sample found during construction that
already meets the residual tolerance ends the solve at once, without completing
the polygon.

For a single residual component, a sign-changing segment is an ordinary scalar
bracket, which for a continuous function certifies a root, so a small
coordinate width ends successfully. For several components, the vertex signs
alone **do not certify that a common root exists**, so a small width is
stagnation, not success; only the residual tolerance ends a multidimensional
solve successfully. The method searches by signs and does not derive a degree
or face certificate from its vertices.

Rectangular systems (`n` different from `m`) use the same recurrence. The
polygon still needs `2**m` sign patterns, and the extra input coordinates are
ordinary coordinates of its geometry. Nothing here guarantees that a root of a
rectangular system exists or is unique.

## Python And Compiled Callbacks

Pass a Python callable, a residual `VectorCallback` for square systems, or a
residual `RectangularCallback` when the input and output lengths differ. Native
code creates them with `make_vector_callback` or `make_rectangular_callback`,
the same callback ABI as the other vector solvers.

A native residual writes every output element and returns zero on success; a
nonzero status raises `NativeCallbackError`. The producer must keep its context
alive through the descriptor's keepalive, must not let exceptions escape, and
must synchronize any shared mutable native state. Reentrant or concurrent use
of one adapter is rejected. Python exceptions propagate as the original
instance.

Setup conversions and copied exports hold the GIL. With native callbacks,
vertex initialization, polygon sampling and the whole numerical `run()` happen
outside the GIL, with no NumPy calls and no per-sample arrays or views. Python
callbacks take the GIL only for their own calls. See
[Compile Python Callbacks](compiling-callbacks.md) for typed Python-syntax and
Cython examples.

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

`h` is a scalar or one finite step per input coordinate. By default every
coordinate uses `max(1, norm_inf(x))`, also at the origin. The search samples
one axis-aligned cube of inputs, then refines the actual sign changes of each
component along its edges to find the missing sign patterns. A complete polygon
keeps one point per pattern with its evaluated residual. The search never
invents signs, does not count zero components as extra patterns, and does not
resample completed vertices before the iteration starts.

Edges are pairs of cube corners that differ in one input coordinate, in any
dimension. Edges whose endpoints have the same sign in a component are skipped
for that component instead of being bisected. The search is bounded and
heuristic: failing to complete a polygon does not prove that no root or
admissible polygon exists. When it fails, supply explicit vertices or change
the starting point or steps.

`polygon_eps` controls the internal sign-change search; the standalone helper
calls it `eps`. The search budgets `max_evaluations`, `max_visits` and
`max_bisections` are separate from the solver's `max_iter`. Resource limits and
finite steps are checked before the native loops start. With `2**n` cube
corners and `2**m` sign patterns, large dimensions are expensive even in native
code.

The standalone helper `compute_admissible_n_polygon` requires a complete
polygon even when its search happens to sample a root; the solver returns such
a root directly.

`cyroot.vector_bracketing` also provides the helpers `get_M`,
`_get_1_simplexes` and `sorted_by_vertices`, which return NumPy arrays. The
solver and the native polygon builder do not use them, or any NumPy operation,
internally.

## Inspection And Resources

`Vrahatis` exposes `step()`, `run()`, `result`, `done`, iteration, `history` and
`history_truncated`. A step is one edge sweep. History stores copies of the
root, residual, polygon, residual polygon, counters and metrics. `history=0`
disables it; `history`, `history_every` and `history_bytes` bound what is kept.
A full history does not stop the solve, and earlier snapshots stay valid after
the solver resumes or is destroyed.

`workspace_bytes` bounds the native workspace and `history_bytes` the native
history. Python conversion objects, exported tuples, allocator metadata and
allocations made by callbacks come on top, so these are not limits on the
process's peak memory. Inputs are copied; caller arrays are never modified.

## Migrating From 1.x

Callbacks receive tuples, so NumPy expressions such as `lambda x: x-target`
need explicit elementwise arithmetic or a conversion of the input. Results
contain tuples, and call counts and iteration paths can differ from 1.x.
Compiled consumers must be rebuilt against the 2.0 declarations. The private
`vrahatis_kernel` and the experimental Mishchenko and Ahmadi methods were
removed. The Eiger–Sikorski–Stenger method, which needs degree and simplex
search machinery, is not implemented.
