# distutils: language=c++
# cython: binding=True
"""Established quasi-Newton routes backed by native solver states."""
import cython
from .utils._function_registering import register
from .native.broyden import _broyden as _native_broyden

cdef extern from 'defaults.hpp' namespace 'cyroot' nogil:
    double ETOL, ERTOL, PTOL, PRTOL
    size_t MAX_ITER

__all__ = ['wolfe_bittner', 'robinson', 'barnes', 'traub_steffensen', 'broyden', 'klement']

@register('cyroot.vector.quasi_newton')
@cython.binding(True)
def wolfe_bittner(F, x0s, F_x0s=None, *, rcond=1e-12,
                  etol=1e-10, ertol=8.881784197001252e-16,
                  ptol=1e-12, prtol=8.881784197001252e-16, max_iter=200,
                  history=0, history_every=1, history_bytes=8*1024*1024,
                  workspace_bytes=64*1024*1024):
    """Native multipoint affine root finding; see docs/native-wolfe-bittner.md."""
    from .native.vector import WolfeBittner
    return WolfeBittner(F, x0s, f_x0s=F_x0s, rcond=rcond,
        etol=etol, ertol=ertol, ptol=ptol, prtol=prtol, max_iter=max_iter,
        history=history, history_every=history_every, history_bytes=history_bytes,
        workspace_bytes=workspace_bytes).run()

#------------------------
# Robinson
#------------------------
@register('cyroot.vector.quasi_newton')
@cython.binding(True)
def robinson(F, x0, x1, F_x0=None, F_x1=None, *, orth='efficient',
             etol=1e-10, ertol=8.881784197001252e-16,
             ptol=1e-12, prtol=8.881784197001252e-16, max_iter=200,
             history=0, history_every=1, history_bytes=8*1024*1024,
             workspace_bytes=64*1024*1024):
    """Run native square Robinson with identity or deterministic Householder probes.

    orth accepts identity/1 or efficient/2; random mode is removed. Probe
    columns and their actual rounded displacements define the divided matrix.
    Options after F_x1 are keyword-only; Python callbacks receive tuples.
    Returns NewtonMethodsReturnType with paired residual/matrix counts.
    Use native Robinson for step/history; see docs/native-robinson.md.
    """
    from .native.vector import Robinson
    return Robinson(F, x0, x1, f_x0=F_x0, f_x1=F_x1, orth=orth,
        etol=etol, ertol=ertol, ptol=ptol, prtol=prtol, max_iter=max_iter,
        history=history, history_every=history_every, history_bytes=history_bytes,
        workspace_bytes=workspace_bytes).run()

#------------------------
# Barnes
#------------------------
@register('cyroot.vector.quasi_newton')
@cython.binding(True)
def barnes(F, x0, F_x0=None, J_x0=None, *,
           etol=1e-10, ertol=8.881784197001252e-16,
           ptol=1e-12, prtol=8.881784197001252e-16, max_iter=200,
           history=0, history_every=1, history_bytes=8*1024*1024,
           workspace_bytes=64*1024*1024):
    """Run native square Barnes with deterministic retained-secant projection.

    J_x0 is a matrix, matching JacobianPlan, or None for automatic central
    initialization. Uses explicit secant residual (old formula=2); formula and
    h are removed and options after J_x0 are keyword-only. Python callbacks
    receive tuples; returns BroydenResult. A direction in the span of retained
    secants reports breakdown without restart. See docs/native-barnes.md.
    """
    from .native.broyden import Barnes
    return Barnes(F, J_x0, x0, f_x0=F_x0,
        etol=etol, ertol=ertol, ptol=ptol, prtol=prtol, max_iter=max_iter,
        history=history, history_every=history_every, history_bytes=history_bytes,
        workspace_bytes=workspace_bytes).run()

################################################################################
# Traub-Steffensen
################################################################################
@register('cyroot.vector.quasi_newton')
@cython.binding(True)
def traub_steffensen(F, x0, F_x0=None, *,
                     etol=1e-10, ertol=8.881784197001252e-16,
                     ptol=1e-12, prtol=8.881784197001252e-16, max_iter=200,
                     history=0, history_every=1, history_bytes=8*1024*1024,
                     workspace_bytes=64*1024*1024):
    """Run native square Traub-Steffensen with residual-dependent differences.

    Each coordinate perturbation is its signed residual component. Differences
    divide by actual rounded displacement. A zero/collapsed perturbation reports
    breakdown without fallback; initial roots skip unused differences. Python
    callbacks receive tuples. Options after F_x0 are keyword-only. Returns
    NewtonMethodsReturnType; use native TraubSteffensen for step/history.
    See docs/native-steffensen.md for the breaking interface changes.
    """
    from .native.vector import TraubSteffensen
    return TraubSteffensen(F, x0, f_x0=F_x0,
        etol=etol, ertol=ertol, ptol=ptol, prtol=prtol, max_iter=max_iter,
        history=history, history_every=history_every, history_bytes=history_bytes,
        workspace_bytes=workspace_bytes).run()

################################################################################
# Broyden
################################################################################
@register('cyroot.vector.quasi_newton')
@cython.binding(True)
def broyden(F, x0, F_x0=None, J_x0=None, *, algo=2,
            etol=ETOL, ertol=ERTOL, ptol=PTOL, prtol=PRTOL,
            max_iter=MAX_ITER, history=0, history_every=1,
            history_bytes=8*1024*1024, workspace_bytes=64*1024*1024):
    """Solve a square system using native good or bad Broyden.

    ``J_x0`` is an initial Jacobian for both variants, or a native
    JacobianPlan/None for numerical initialization. Select exact built-in
    1/'good' or 2/'bad' (default). Python callbacks receive copied tuples.
    Returns BroydenResult with a copied secant approximation and diagnostics.
    Use native BroydenGood/BroydenBad states to inspect intermediate steps.
    See docs/native-broyden.md for the breaking interface changes.
    """
    return _native_broyden(F, x0, F_x0, J_x0, algo, etol, ertol, ptol, prtol,
                          max_iter, history, history_every, history_bytes,
                          workspace_bytes)

################################################################################
# Klement
################################################################################
@register('cyroot.vector.quasi_newton')
@cython.binding(True)
def klement(F, x0, F_x0=None, J_x0=None, *,
            etol=1e-10, ertol=8.881784197001252e-16,
            ptol=1e-12, prtol=8.881784197001252e-16, max_iter=200,
            history=0, history_every=1, history_bytes=8*1024*1024,
            workspace_bytes=64*1024*1024):
    """Run native square Klement with row-weighted secant Jacobian updates.

    J_x0 is a matrix, a matching native JacobianPlan, or None for automatic
    central initialization. Python residuals receive tuples. Options after
    J_x0 are keyword-only and h is removed. Singular solves or zero row update
    denominators report breakdown; root success requires residual convergence.
    Returns BroydenResult; use native Klement for step/history inspection.
    See docs/native-klement.md for the breaking interface changes.
    """
    from .native.broyden import Klement
    return Klement(F, J_x0, x0, f_x0=F_x0,
        etol=etol, ertol=ertol, ptol=ptol, prtol=prtol, max_iter=max_iter,
        history=history, history_every=history_every, history_bytes=history_bytes,
        workspace_bytes=workspace_bytes).run()
