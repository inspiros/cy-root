# Native Higher Mixed Partial Derivatives

`TensorDifferencePlan` estimates dense derivatives of order three and above for
functions with independent input and output dimensions. It uses native C++
sampling, rounded-node finite-difference weights and tensor arithmetic. Compiled
callbacks run without the GIL. Python callbacks acquire it for each call; plan
construction and copied result exports also use it. No NumPy or SymPy calls occur
inside the new plan or its kernel. NumPy and SciPy remain package dependencies.

A partial derivative differentiates with respect to one input. A mixed partial
uses multiple inputs, and its order counts all differentiations. For example,
\(\partial^3F/\partial x^2\partial y\) differentiates twice in \(x\) and once in
\(y\). If \(F(x,y)=x^2y\), this derivative is \(2\). A third derivative tensor
for a scalar-valued two-input function has eight entries, with four distinct
symmetric components: `xxx`, `xxy`, `xyy` and `yyy`. Derivative order is separate
from a root solver's convergence order.

```python
from cyroot.native import TensorDifferencePlan

def function(point):
    x, y = point
    return (x*x*y, x*y*y)

plan = TensorDifferencePlan(2, 2, 3, step=.125)
answer = plan.evaluate(function, (.5, 1.))
assert answer.status == 'ok'
assert plan.shape == (2, 2, 2, 2)
assert abs(answer.derivative[0][0][0][1]-2.) < 1e-10
assert abs(answer.derivative[1][0][1][1]-2.) < 1e-10
cached = plan.evaluate(function, (.5, 1.), f_x=function((.5, 1.)))
assert cached.f_calls == answer.f_calls-1
assert cached.derivative == answer.derivative
```

## Shapes And Usage

The constructor is
`TensorDifferencePlan(input_dimension, output_dimension, order, scheme='central',
step=None, *, max_nodes=65, max_terms=1_000_000, memory_bytes=64*1024*1024)`.
Dimensions are positive C integers; order is at least three. Use `JacobianPlan`
or `RectangularJacobianPlan` for order one, and `HessianPlan` or
`RectangularHessianPlan` for order two. The 1.x derivative wrappers were removed. Use these native plans directly;
see [Finite Differences](finite-differences.md) for the replacement mapping.
Standalone estimates return copied nested tuples.

For \(n\) inputs and \(m\) outputs, the copied nested tuple has shape
\((m,n,\ldots,n)\), with `order` input axes. For instance,
`derivative[o][i][j][k]` estimates the third partial of residual `o` with respect
to input coordinates `i`, `j`, `k`. Repeated indices represent repeated
 differentiation. Each symmetric component is computed once and copied to all
its permutations, giving bit-identical symmetric entries. This assumes the
smoothness needed for commuting mixed derivatives; symmetry does not certify
that a nonsmooth function has such derivatives.

`evaluate(function, point, f_x=None)` accepts an ordinary Python function,
a matching residual-kind `RectangularCallback`, or a square `VectorCallback`
when both dimensions agree. Inputs passed to Python callbacks are copied tuples.
The optional cached base residual avoids one call. Each unique stencil point is
sampled once per evaluation, including points shared by different tensor entries.
No samples are reused across separate evaluations. Plans are immutable and their
native graph can be reused with separate caller-owned scratch.

The result is `TensorDerivativeEstimate(derivative, f_calls, status, callback_code)`.
`f_calls` counts attempted residual calls, including the base unless supplied.
Statuses are `ok`, `nonfinite`, `breakdown`, or `callback_failure`. Collapsed
rounded stencil points report `breakdown`; nonfinite input, output or arithmetic
reports `nonfinite`. Failure poisons the complete tensor with NaNs. Original
Python callback exceptions are re-raised; native callback failures raise
`NativeCallbackError` carrying the estimate and stage `'function'` for the base
or `'derivative'` for a stencil sample. Adapter reservations include conversion,
execution, error delivery and result boxing.

## Stencil And Numerical Limits

A component with multiplicity \(r\) on one axis uses `r+1` nodes for forward or
backward differences. Central differences use `r+1+(r%2)` nodes, including the
base and requested offsets `+1,-1,+2,-2,...`. Tensor-product terms use these
per-axis stencils. Setup enumerates only symmetric components and their needed
points, avoiding a full grid over every input axis.

`step` is `None`, a finite nonzero real scalar or a copied length-\(n\) vector;
signed steps are supported. Automatic per-axis steps are
\(\max(1,|x_j|)\epsilon^{1/(k+a)}\), with total derivative order \(k\),
\(a=2\) for central and \(a=1\) for one-sided differences. This is a heuristic,
not an error estimate. On ideal symmetric nodes, central stencils have nominal
second-order truncation error; one-sided stencils have nominal first-order error.
Actual rounding can reduce that symmetry and alter the error behavior.

Weights use Fornberg's recurrence on normalized **actual rounded displacements**.
The kernel subtracts the base residual before compensated summation and rescales
with binary exponents to avoid premature range loss in products of step scales.
Final underflow can round to zero; overflow and nonfinite intermediate arithmetic
fail explicitly. Very high orders, noisy functions, cancellation and
ill-conditioned stencils can produce inaccurate finite results. Raising a
resource cap does not improve numerical precision.

## Budgets And Native Consumers

Dense output contains \(mn^k\) doubles, so cost grows exponentially with order.
`max_nodes` limits the largest per-axis stencil; the default allows order 64.
`max_terms` bounds total tensor-product visits before enumeration, counting
repeated visits to shared samples. The sample cache itself stores unique points.
Requests exceeding shape, term or byte limits fail before residual sampling.

`memory_bytes` limits the logical native graph and step spans plus one native
evaluation's input, base, output, cache and scratch. Mandatory graph counts and
a cache-entry cap are computed before graph enumeration; each new entry is
checked before insertion. Native vector capacity/allocator overhead, temporary
Python planning objects, copied result tuples and callback internals use extra
memory. This is a native storage budget, not a bound on process peak memory.

Read-only properties expose `input_dimension`, `output_dimension`, `order`,
`scheme`, copied `step`, `shape`, unique nonbase `sample_count`, `storage_bytes`,
`workspace_size`, `workspace_bytes`, `output_bytes`, `evaluation_bytes` and
`memory_bytes`. Storage plus evaluation bytes equals reported memory bytes.
No native numerical workspace allocation or growth occurs inside the kernel.

Cython consumers cimport `TensorDifferencePlan` from `cyroot.native.higher_mixed`
and `TensorDifferenceConfig`/`compute_tensor_difference` from
`cyroot.native.higher_mixed_ops`. An initialized plan's `copy_config` lends
immutable graph/step pointers: the plan must remain pinned. `compute` and the
free kernel accept disjoint raw input/base/output/scratch spans and caller-owned
attempt/status counters; capacities count doubles. Native output is packed as
`out[o+m*flat]`, with the first input index varying fastest. Graph contents and
callback pointer lifetimes are trusted caller contracts. Invalid declared
configuration or insufficient capacities return `breakdown` before output or
scratch writes, resetting only callback code. Valid-span failures poison output
and restore the scratch trial point. The callback must not throw or mutate its
borrowed point. No new callback descriptor ABI is introduced.
