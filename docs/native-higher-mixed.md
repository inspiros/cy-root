# Native Higher Mixed Partial Derivatives

`TensorDifferencePlan` estimates dense derivatives of order three and above
for functions with any input and output dimensions. It uses native C++
sampling, finite-difference weights on the rounded nodes and tensor
arithmetic, with no NumPy or SymPy calls. Compiled callbacks run without the
GIL; Python callbacks take it for each call, as do plan construction and result
export.

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
`RectangularHessianPlan` for order two; [Finite Differences](finite-differences.md)
maps the removed 1.x derivative wrappers to these plans. Estimates are copied
nested tuples.

For \(n\) inputs and \(m\) outputs, the copied nested tuple has shape
\((m,n,\ldots,n)\), with `order` input axes. For instance,
`derivative[o][i][j][k]` estimates the third partial of residual `o` with respect
to input coordinates `i`, `j`, `k`; repeated indices mean repeated
differentiation. Each distinct component is computed once and copied to all of
its permutations, so symmetric entries are bit-identical. This assumes the
smoothness under which mixed derivatives commute; the symmetry of the result
says nothing about whether a nonsmooth function has such derivatives.

`evaluate(function, point, f_x=None)` accepts a Python function, a residual
`RectangularCallback` of matching shape, or a square `VectorCallback` when the
dimensions are equal. Python callbacks receive copied tuples. A cached base
residual saves one call. Each distinct stencil point is sampled once per
evaluation, even when several tensor entries use it, but nothing is reused
between evaluations. Plans are immutable, and their native graph can be reused
with separate caller-owned scratch.

The result is `TensorDerivativeEstimate(derivative, f_calls, status, callback_code)`.
`f_calls` counts attempted residual calls, including the base unless supplied.
The status is `ok`, `nonfinite`, `breakdown` or `callback_failure`. Stencil
points that collapse after rounding report `breakdown`, and nonfinite inputs,
outputs or arithmetic `nonfinite`; any failure fills the whole tensor with NaN.
Python callback exceptions are re-raised as they are; native callback failures
raise `NativeCallbackError` with the estimate and the stage `'function'` for
the base or `'derivative'` for a stencil sample. Adapters stay reserved through
conversion, execution, error delivery and result construction.

## Stencil And Numerical Limits

A component with multiplicity \(r\) on one axis uses `r+1` nodes for forward or
backward differences. Central differences use `r+1+(r%2)` nodes, including the
base and requested offsets `+1,-1,+2,-2,...`. Tensor-product terms use these
per-axis stencils. Setup enumerates only the distinct components and the
points they need, rather than a full grid over every input axis.

`step` is `None`, a finite nonzero real number, or a vector of \(n\) such
numbers, which is copied; negative steps are allowed. Automatic steps are
\(\max(1,|x_j|)\epsilon^{1/(k+a)}\) per axis, where \(k\) is the total
derivative order and \(a\) is 2 for central and 1 for one-sided differences.
This is a rule of thumb, not an error estimate. On exactly symmetric nodes,
central stencils have second-order and one-sided stencils first-order
truncation error; rounding can break that symmetry and change the error.

The weights come from Fornberg's recurrence on the normalized **actual rounded
displacements**. The kernel subtracts the base residual before a compensated
sum and rescales with binary exponents, so that products of step scales do not
lose range prematurely. A result can still underflow to zero, while overflow
and nonfinite intermediate arithmetic fail explicitly. Very high orders, noisy
functions, cancellation and ill-conditioned stencils can give inaccurate
finite results, and raising a resource limit does not improve precision.

## Budgets And Native Consumers

Dense output contains \(mn^k\) doubles, so cost grows exponentially with order.
`max_nodes` limits the largest per-axis stencil; the default allows order 64.
`max_terms` bounds total tensor-product visits before enumeration, counting
repeated visits to shared samples. The sample cache itself stores unique points.
Requests exceeding shape, term or byte limits fail before residual sampling.

`memory_bytes` limits the native graph and steps plus one evaluation's input,
base, output, cache and scratch. The required graph sizes and a cap on cache
entries are computed before the graph is enumerated, and each new entry is
checked before it is inserted. Native vector capacity and allocator overhead,
temporary Python planning objects, copied result tuples and callback internals
come on top, so this budget does not bound the process's peak memory.

The read-only properties are `input_dimension`, `output_dimension`, `order`,
`scheme`, the copied `step`, `shape`, `sample_count` (distinct points other
than the base), `storage_bytes`, `workspace_size`, `workspace_bytes`,
`output_bytes`, `evaluation_bytes` and `memory_bytes`, which is the sum of
`storage_bytes` and `evaluation_bytes`. The kernel never allocates or grows
workspace.

Cython code cimports `TensorDifferencePlan` from `cyroot.native.higher_mixed`
and `TensorDifferenceConfig` and `compute_tensor_difference` from
`cyroot.native.higher_mixed_ops`. An initialized plan's `copy_config` lends out
its immutable graph and step pointers, so the plan must stay alive. `compute`
and the free kernel take non-overlapping raw input, base, output and scratch
spans and caller-owned attempt and status counters, with capacities in doubles.
The output is packed as `out[o+m*flat]`, with the first input index varying
fastest. The caller is responsible for the graph contents and for keeping the
callback's pointers valid. An invalid configuration or too little capacity
returns `breakdown` before anything is written, resetting only the callback
code. With valid spans, any failure fills the output with NaN and restores the
scratch trial point. The callback must not throw or modify the borrowed point.
