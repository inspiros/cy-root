# CyRoot Documentation

<div class="cyroot-intro">
<p class="eyebrow">CYTHON ROOT FINDING · VERSION 2.0</p>
<p class="lead">Where Functions Meet Zero</p>
<p>Start with a Python function. Explore every iteration. Move the numerical hot path into native code when you need it.</p>
</div>

![CyRoot: Recorded Newton Tangents And A Dogleg Path](../assets/hero.svg)

<div class="cyroot-paths">
<a class="path-card" href="quick-start.md"><span class="card-number">01 / START</span><strong>Solve Your First Root</strong><span>Scalar equations, nonlinear systems, and inspectable steps in a few lines.</span></a>
<a class="path-card" href="choosing-methods.md"><span class="card-number">02 / EXPLORE</span><strong>Choose Your Method</strong><span>Match the algorithm to your derivatives, system shape, and storage budget.</span></a>
<a class="path-card" href="compiling-callbacks.md"><span class="card-number">03 / COMPILE</span><strong>Take The Native Path</strong><span>Typed Cython callbacks, owned buffers, and numerical loops without the GIL.</span></a>
</div>

## A Small Function, A Native Engine

```python
from cyroot import newton

answer = newton(lambda x: x*x - 2., None, 1.5)
assert answer.success
assert abs(answer.root - 2.**.5) < 1e-10
```

Pass an analytic derivative, or `None` for a native finite difference. Vector
callbacks may return tuples, lists or NumPy arrays. With native callbacks the
whole numerical loop runs without the GIL; Python callbacks take it for each
call. NumPy and SciPy are required.

Version 2.0 is not backward compatible with 1.x; the
[migration guides](scalar-native-migration.md) map the old API to the new one.
[Stopping And Results](stopping.md) explains how to tell a residual root from a
narrow scalar bracket or a stationary point with a nonzero residual.

## Explore The Manual

### Getting Started

- [Quick Start](quick-start.md)
- [Choose A Method](choosing-methods.md)
- [Building And Testing](building.md)

### Scalar Solvers

- [Newton](native-newton.md)
- [Secant](native-secant.md)
- [Halley Family](native-higher-order.md)
- [Householder](native-householder.md)

### Vector Solvers

- [Vector Newton](native-vector.md)
- [Vrahatis Bisection](native-vrahatis.md)
- [Broyden](native-broyden.md)
- [Barnes](native-barnes.md)
- [Klement](native-klement.md)
- [Robinson](native-robinson.md)
- [Traub–Steffensen](native-steffensen.md)
- [Wolfe–Bittner](native-wolfe-bittner.md)
- [Vector Chebyshev](native-vector-chebyshev.md)
- [Vector Halley And Super-Halley](native-vector-halley.md)
- [Vector Modified Halley](native-vector-modified.md)
- [Rectangular Chebyshev](native-rectangular-chebyshev.md)
- [Rectangular Halley](native-rectangular-halley.md)
- [Rectangular Modified Halley](native-rectangular-modified.md)
- [Native Least Squares](native-least-squares.md)

### Matrix-Free And Optimization

- [Damped Newton](native-damped-newton.md)
- [Jacobian Products](jacobian-vector-products.md)
- [Newton–Krylov](native-newton-krylov.md)
- [Optimization And Objective Derivatives](optimization.md)
- [Levenberg–Marquardt](native-levenberg-marquardt.md)

### Derivatives

- [Numerical Derivatives](numerical-derivatives.md)
- [Finite Differences](finite-differences.md)
- [Scalar First Derivatives](native-scalar-derivative.md)
- [Scalar Second Derivatives](native-second-derivative.md)
- [Runtime-Order Scalar Derivatives](native-higher-derivative.md)
- [Jacobian Plans](native-jacobian.md)
- [Vector Hessians](native-hessian.md)
- [Rectangular Hessians](native-rectangular-hessian.md)
- [Higher Mixed Partials](native-higher-mixed.md)

### Callbacks And Native API

- [Native Core And Ownership](native-core.md)
- [Compile Python Callbacks](compiling-callbacks.md)
- [Rectangular Callbacks](native-rectangular-callbacks.md)
- [Tensor Callbacks And Contractions](native-rectangular-tensors.md)

### Stopping And Results

- [Stopping And Results](stopping.md)

### Migration From 1.x

- [Scalar Migration](scalar-native-migration.md)
- [Higher-Order Vector Migration](vector-higher-order-migration.md)

The [repository README](../README.md) includes the algorithm gallery and project links.
