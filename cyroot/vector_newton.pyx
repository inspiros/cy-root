# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False


from cyroot.defaults cimport ETOL, ERTOL, PTOL, PRTOL, MAX_ITER
cimport cython

from .utils._function_registering import register

__all__ = [
    'generalized_newton',
    'generalized_damped_newton',
    'generalized_halley',
    'generalized_super_halley',
    'generalized_chebyshev',
    'generalized_tangent_hyperbolas',
]

################################################################################
# Generalized Newton
################################################################################
@register('cyroot.vector.newton')
@cython.binding(True)
def generalized_newton(F, J, x0, F_x0=None, J_x0=None, *,
                       etol=ETOL, ertol=ERTOL,
                       ptol=PTOL, prtol=PRTOL, max_iter=MAX_ITER,
                       history=0, history_every=1, history_bytes=8*1024*1024,
                       workspace_bytes=64*1024*1024):
    """Run native Newton with an analytic or numerical Jacobian.

    Point and residual lengths infer the Jacobian shape. J is a Python/native
    callback, a matching JacobianPlan/RectangularJacobianPlan or None for central
    differences. Python callbacks receive copied tuples. Square steps use LU;
    rectangular steps use rank-aware least squares. Singular square systems
    report breakdown. Only residual convergence reports root success. Options
    after J_x0 are keyword-only; h and legacy providers are removed.
    Use VectorNewton for step/run/history/rank; see docs/native-vector.md.
    """
    from .native.vector import VectorNewton
    return VectorNewton(F, J, x0, f_x0=F_x0, jac_x0=J_x0,
        etol=etol, ertol=ertol, ptol=ptol, prtol=prtol, max_iter=max_iter,
        history=history, history_every=history_every, history_bytes=history_bytes,
        workspace_bytes=workspace_bytes).run()

@register('cyroot.vector.newton')
@cython.binding(True)
def generalized_damped_newton(F, J, x0, F_x0=None, J_x0=None, *,
                              c1=1e-4, contraction=.5, max_line_search=32, **options):
    """Armijo-backtracked Newton with the same shape inference as generalized_newton."""
    from .native.damped_newton import VectorDampedNewton
    return VectorDampedNewton(F, J, x0, f_x0=F_x0, jac_x0=J_x0, c1=c1,
                              contraction=contraction, max_line_search=max_line_search, **options).run()


################################################################################
# Higher-order methods: established names, native shared engine
################################################################################
cdef object _higher_order_solver(object F, object J, object H, object x0,
                                 object F_x0, object J_x0, object H_x0,
                                 int method, object alpha, object options):
    from .native.vector_chebyshev import (
        VectorHalley, VectorModifiedHalley, VectorSuperHalley, VectorChebyshev,
    )
    if method == 0:
        if alpha is None:
            solver_type = VectorHalley
        else:
            options['alpha'] = alpha
            solver_type = VectorModifiedHalley
    elif method == 1:
        solver_type = VectorSuperHalley
    else:
        solver_type = VectorChebyshev
    return solver_type(F, J, H, x0, f_x0=F_x0, jac_x0=J_x0,
                       hess_x0=H_x0, **options).run()


@register('cyroot.vector.newton')
@cython.binding(True)
def generalized_halley(F, J, H, x0, F_x0=None, J_x0=None, H_x0=None, *,
                       alpha=None, etol=ETOL, ertol=ERTOL,
                       ptol=PTOL, prtol=PRTOL, max_iter=MAX_ITER,
                       history=0, history_every=1, history_bytes=8*1024*1024,
                       workspace_bytes=64*1024*1024):
    """Run native matrix Halley, or modified Halley with an explicit alpha.

    Breaking change: the default replaces the old componentwise vector update.
    All alpha paths support specified rectangular least-squares corrections.
    Exactly 0/.5/1 retain their distinct normalized formulas. Singular required
    square systems report breakdown. Python
    callbacks receive tuples. None J/H select native numerical plans; explicit
    JacobianPlan/HessianPlan configure them. Legacy h is removed. Options and
    caches are normalized under native callback reservations. See
    docs/vector-higher-order-migration.md; use native classes for step/history.
    """
    return _higher_order_solver(F, J, H, x0, F_x0, J_x0, H_x0, 0, alpha,
        dict(etol=etol, ertol=ertol, ptol=ptol, prtol=prtol, max_iter=max_iter,
             history=history, history_every=history_every, history_bytes=history_bytes,
             workspace_bytes=workspace_bytes))


@register('cyroot.vector.newton')
@cython.binding(True)
def generalized_super_halley(F, J, H, x0, F_x0=None, J_x0=None, H_x0=None, *,
                             etol=ETOL, ertol=ERTOL,
                             ptol=PTOL, prtol=PRTOL, max_iter=MAX_ITER,
                             history=0, history_every=1, history_bytes=8*1024*1024,
                             workspace_bytes=64*1024*1024):
    """Run native matrix Super-Halley with square LU or rectangular least squares.

    Equivalent update to generalized_halley(alpha=1). None J/H choose native
    plans; square LU remains strict. Rectangles use rank-aware minimum-norm
    corrections. See docs/native-rectangular-halley.md for the explicit policy.
    """
    return _higher_order_solver(F, J, H, x0, F_x0, J_x0, H_x0, 1, None,
        dict(etol=etol, ertol=ertol, ptol=ptol, prtol=prtol, max_iter=max_iter,
             history=history, history_every=history_every, history_bytes=history_bytes,
             workspace_bytes=workspace_bytes))


@register('cyroot.vector.newton')
@cython.binding(True)
def generalized_chebyshev(F, J, H, x0, F_x0=None, J_x0=None, H_x0=None, *,
                         etol=ETOL, ertol=ERTOL,
                         ptol=PTOL, prtol=PRTOL, max_iter=MAX_ITER,
                         history=0, history_every=1, history_bytes=8*1024*1024,
                         workspace_bytes=64*1024*1024):
    """Run native Chebyshev with square LU or rectangular least-squares steps.

    Equivalent update to generalized_halley(alpha=0). None J/H choose matching
    native plans. Rectangular dimensions are inferred; rank-deficient least-
    squares corrections are allowed, and only residual convergence is success.
    Square LU remains strict. See docs/native-rectangular-chebyshev.md.
    """
    return _higher_order_solver(F, J, H, x0, F_x0, J_x0, H_x0, 2, None,
        dict(etol=etol, ertol=ertol, ptol=ptol, prtol=prtol, max_iter=max_iter,
             history=history, history_every=history_every, history_bytes=history_bytes,
             workspace_bytes=workspace_bytes))


@register('cyroot.vector.newton')
@cython.binding(True)
def generalized_tangent_hyperbolas(F, J, H, x0, F_x0=None, J_x0=None, H_x0=None, *,
                                  etol=ETOL, ertol=ERTOL,
                                  ptol=PTOL, prtol=PRTOL, max_iter=MAX_ITER,
                                  history=0, history_every=1, history_bytes=8*1024*1024,
                                  workspace_bytes=64*1024*1024):
    """Run native matrix Halley, including rectangular least-squares corrections.

    The two old square formulas are equivalent; formula and h are removed.
    Python callbacks receive tuples. None J/H choose native plans. Singular
    square systems report breakdown, without least-squares fallback.
    See docs/vector-higher-order-migration.md for the breaking interface changes.
    """
    return _higher_order_solver(F, J, H, x0, F_x0, J_x0, H_x0, 0, None,
        dict(etol=etol, ertol=ertol, ptol=ptol, prtol=prtol, max_iter=max_iter,
             history=history, history_every=history_every, history_bytes=history_bytes,
             workspace_bytes=workspace_bytes))
