# Changelog

## 2.0.0 — Unreleased

- Add scalar and vector damped Newton with native Armijo backtracking, including rectangular systems and resumable accepted-step histories.

A breaking overhaul of the numerical core, callback interfaces, and solver
inspection. NumPy and SciPy remain dependencies; native solver loops use
preallocated storage and can run without the GIL when given native callbacks.
Python callbacks still acquire the GIL.

### Breaking Changes

- Established scalar and vector entry points now use the native engines. Vector
  callbacks receive tuples; compatible sequence and NumPy outputs are accepted
  at Python boundaries. Result snapshots use immutable tuples.
- Numerical controls are keyword-only. Legacy `h`, `formula`, and `c_code`
  options are removed where replaced by explicit native derivative plans or
  method selectors. Compiled consumers must rebuild against the new declarations.
  A step `h` that no numerical derivative would use is rejected. The redundant
  `bisect(algo=...)` selector is removed; native bisection is sign-only.
- Every dispatcher entry takes `(function, derivative_or_jacobian, x0)`, including
  the scalar and vector Levenberg–Marquardt, BFGS, L-BFGS and Powell dogleg
  conveniences; pass `None` for numerical derivatives.
- Vector Halley-family methods use coupled matrix corrections instead of the
  old componentwise formulas. Generalized Newton and the higher-order vector
  methods support rectangular systems through rank-aware least-squares steps;
  Broyden and several other quasi-Newton methods require square systems.
- Singular square solves report breakdown instead of silently substituting a
  pseudoinverse. Small steps with an unsatisfied residual report stagnation.
  Bracket convergence and residual convergence have distinct stopping reasons.
- Finite-difference evaluation counts change because duplicate and unused samples
  are removed and numerical derivative samples are accounted for.
- Unverified Ahmadi and experimental Mishchenko remain outside ordinary exports
  and method discovery.

### Native Solvers And Callbacks

- Route all 26 supported scalar methods and the supported vector methods through
  native numerical loops, including bracketing, interpolation, Vrahatis, and the
  quasi-Newton family.
- Replace symbolic Householder formulas with a runtime-order scalar recurrence
  based on reciprocal Taylor coefficients; remove the SymPy dependency.
- Add scalar and vector Levenberg–Marquardt, BFGS, L-BFGS, and Powell dogleg.
  Optimization-based methods check the original residual before reporting a root;
  a stationary point with a nonzero residual is not root convergence. BFGS and
  L-BFGS measure stationarity relative to the current residual and Jacobian, so
  residual scaling and multiple roots do not stop them early. `max_iter=0` or
  `None` removes their iteration limit, as for the root solvers.
- Point, residual and Jacobian inputs that are real buffers other than native
  float64 (for example integer or float32 NumPy arrays) are converted element by
  element; complex and boolean buffers are rejected.
- Accept Python functions and native Cython/C/C++ callbacks with pinned owners,
  explicit native statuses, and propagation of the original Python exception.
- Add owned native buffers, borrowed signed-stride views, reusable solver scratch,
  checked workspace budgets, and guarded callback/solver reentry.

### Derivatives And Objective Adapters

- Add native scalar finite-difference plans and square/rectangular Jacobian,
  Hessian, and higher mixed-partial tensor plans. Use actual rounded sample nodes,
  cache reusable samples, and bound stencil/storage requirements.
- Migrate `GeneralizedFiniteDifference` onto the native plans while retaining its
  NumPy-compatible Python boundaries.
- Add `ComplexStepDerivative` and `ComplexStepJacobian` for holomorphic complex
  implementations of real residuals.
- Add `AdaptiveDerivative`, `AdaptiveSecondDerivative`, `AdaptiveJacobian`, and
  `AdaptiveHessian` with bounded Richardson refinement, copied error estimates,
  and rejection of unconverged estimates during solver use. Estimated errors are
  diagnostic estimates, not rigorous bounds.
- Add exact least-squares objective value, gradient, Hessian, and higher derivative
  transforms, plus a bridge for external optimizers with separate root checks.

### Iteration Inspection And Examples

- Add resumable `step()`/`run()` solvers, bounded copied iteration histories,
  per-stage evaluation diagnostics, and explicit termination reasons.
- Add `RootSnapshot` and `as_root_snapshot` for common single-root inspection,
  method discovery, and scalar/vector optimization conveniences.
- Add compiled callback examples and reusable Matplotlib rendering helpers, with
  scalar/vector animations and an optional 3D residual-norm surface view.
- Refresh the README and CyRoot logo, with a generated SVG hero using recorded
  Newton and Powell dogleg histories. Keep public images in `assets/`; package
  descriptions pin their image URLs to the matching release tag.

### Numerical Corrections

- Correct scalar and vector finite-difference weights, including mixed partials,
  and validate sample geometry, shapes, finite outputs, and resource limits.
- Correct scalar bracket bookkeeping, Muller's quadratic step, Sidi's
  zero-derivative behavior, TOMS748 differences, and scaled Wu interpolation.
- Correct bad Broyden's inverse update, Wolfe–Bittner consistency checks, Robinson
  direction orientation, and deterministic Barnes secant projections.
- Correct Vrahatis cube-edge discovery and require residual convergence rather
  than accepting a narrow vertex geometry alone.
- Reject nonfinite LU factors even when LAPACK reports success. Validate linear
  algebra capacities, preserve SVD inputs, and check LAPACK failure statuses.
- Use fixed initial residual scales for relative tolerances and current iterate
  scales for step tolerances. Preserve terminal callback-failure diagnostics and
  reserve adapters during setup, stepping, and result/history export.
- Honor interval-splitting budgets and report both per-root and overall status.

### Build And Distribution

- Target CPython 3.11–3.14, Cython 3.3.0, and C++17. Require NumPy 2.x and SciPy
  1.13–1.17; SciPy supplies the native BLAS/LAPACK interfaces.
- Centralize metadata in `pyproject.toml`. Build authored Cython sources with
  generated C++ under `build/`; remove stale generated sources and unused
  compiled experiments. Include declarations, type stubs, and native headers.
- Run installed tests in isolated subprocesses with bounded parallelism and
  import-provenance checks. Cover minimum dependencies in the CI matrix.
- Trigger checks/builds on version-tag pushes. Publish to PyPI only through the
  manual Publish workflow, after matching-version builds and installed tests.
