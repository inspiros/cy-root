# distutils: language=c++
# cython: language_level=3
"""Established scalar bracketing names backed by native state machines."""
import cython
from .utils._function_registering import register

__all__ = ['bisect','hybisect','regula_falsi','illinois','pegasus','anderson_bjorck',
           'dekker','brent','chandrupatla','ridders','toms748','wu','itp']

@register('cyroot.scalar.bracketing')
@cython.binding(True)
def bisect(f,a,b,f_a=None,f_b=None,*,
           etol=1e-10,ertol=8.881784197001252e-16,
           ptol=1e-12,prtol=8.881784197001252e-16,max_iter=200,**options):
    """Native sign-only bisection of a valid bracket."""
    from .native.scalar import bisect as solve
    return solve(f,a,b,f_a=f_a,f_b=f_b,etol=etol,ertol=ertol,ptol=ptol,
                 prtol=prtol,max_iter=max_iter,**options)


@register('cyroot.scalar.bracketing')
@cython.binding(True)
def hybisect(f,interval_f,a,b,*, etol=1e-10,ertol=8.881784197001252e-16,
             ptol=1e-12,prtol=8.881784197001252e-16,max_iter=200,
             max_split_iter=200,**options):
    """Native bounded interval search; use Hybisect for event/history inspection."""
    from .native.hybisect import hybisect as solve
    return solve(f,interval_f,a,b,etol=etol,ertol=ertol,ptol=ptol,prtol=prtol,
                 max_iter=max_iter,max_split_iter=max_split_iter,**options)


@register('cyroot.scalar.bracketing')
@cython.binding(True)
def regula_falsi(f,a,b,f_a=None,f_b=None,
           *, etol=1e-10,ertol=8.881784197001252e-16,
           ptol=1e-12,prtol=8.881784197001252e-16,max_iter=200,**options):
    """Run native regula_falsi; use cyroot.native.BracketSolver for stepping/history."""
    from .native.bracketing import regula_falsi as solve
    return solve(f,a,b,f_a=f_a,f_b=f_b,
                 etol=etol,ertol=ertol,ptol=ptol,prtol=prtol,max_iter=max_iter,**options)


@register('cyroot.scalar.bracketing')
@cython.binding(True)
def illinois(f,a,b,f_a=None,f_b=None,
           *, etol=1e-10,ertol=8.881784197001252e-16,
           ptol=1e-12,prtol=8.881784197001252e-16,max_iter=200,**options):
    """Run native illinois; use cyroot.native.BracketSolver for stepping/history."""
    from .native.bracketing import illinois as solve
    return solve(f,a,b,f_a=f_a,f_b=f_b,
                 etol=etol,ertol=ertol,ptol=ptol,prtol=prtol,max_iter=max_iter,**options)


@register('cyroot.scalar.bracketing')
@cython.binding(True)
def pegasus(f,a,b,f_a=None,f_b=None,
           *, etol=1e-10,ertol=8.881784197001252e-16,
           ptol=1e-12,prtol=8.881784197001252e-16,max_iter=200,**options):
    """Run native pegasus; use cyroot.native.BracketSolver for stepping/history."""
    from .native.bracketing import pegasus as solve
    return solve(f,a,b,f_a=f_a,f_b=f_b,
                 etol=etol,ertol=ertol,ptol=ptol,prtol=prtol,max_iter=max_iter,**options)


@register('cyroot.scalar.bracketing')
@cython.binding(True)
def anderson_bjorck(f,a,b,f_a=None,f_b=None,
           *, etol=1e-10,ertol=8.881784197001252e-16,
           ptol=1e-12,prtol=8.881784197001252e-16,max_iter=200,**options):
    """Run native anderson_bjorck; use cyroot.native.BracketSolver for stepping/history."""
    from .native.bracketing import anderson_bjorck as solve
    return solve(f,a,b,f_a=f_a,f_b=f_b,
                 etol=etol,ertol=ertol,ptol=ptol,prtol=prtol,max_iter=max_iter,**options)


@register('cyroot.scalar.bracketing')
@cython.binding(True)
def dekker(f,a,b,f_a=None,f_b=None,
           *, etol=1e-10,ertol=8.881784197001252e-16,
           ptol=1e-12,prtol=8.881784197001252e-16,max_iter=200,**options):
    """Run native dekker; use cyroot.native.BracketSolver for stepping/history."""
    from .native.bracketing import dekker as solve
    return solve(f,a,b,f_a=f_a,f_b=f_b,
                 etol=etol,ertol=ertol,ptol=ptol,prtol=prtol,max_iter=max_iter,**options)


@register('cyroot.scalar.bracketing')
@cython.binding(True)
def ridders(f,a,b,f_a=None,f_b=None,
           *, etol=1e-10,ertol=8.881784197001252e-16,
           ptol=1e-12,prtol=8.881784197001252e-16,max_iter=200,**options):
    """Run native ridders; use cyroot.native.BracketSolver for stepping/history."""
    from .native.bracketing import ridders as solve
    return solve(f,a,b,f_a=f_a,f_b=f_b,
                 etol=etol,ertol=ertol,ptol=ptol,prtol=prtol,max_iter=max_iter,**options)


@register('cyroot.scalar.bracketing')
@cython.binding(True)
def wu(f,a,b,f_a=None,f_b=None,
           *, etol=1e-10,ertol=8.881784197001252e-16,
           ptol=1e-12,prtol=8.881784197001252e-16,max_iter=200,**options):
    """Run native wu; use cyroot.native.BracketSolver for stepping/history."""
    from .native.bracketing import wu as solve
    return solve(f,a,b,f_a=f_a,f_b=f_b,
                 etol=etol,ertol=ertol,ptol=ptol,prtol=prtol,max_iter=max_iter,**options)


@register('cyroot.scalar.bracketing')
@cython.binding(True)
def brent(f,a,b,f_a=None,f_b=None,*, interp_method='hyperbolic',sigma=1e-5,
           etol=1e-10,ertol=8.881784197001252e-16,
           ptol=1e-12,prtol=8.881784197001252e-16,max_iter=200,**options):
    """Run native brent; use cyroot.native.BracketSolver for stepping/history."""
    from .native.bracketing import brent as solve
    return solve(f,a,b,f_a=f_a,f_b=f_b,interp_method=interp_method,sigma=sigma,
                 etol=etol,ertol=ertol,ptol=ptol,prtol=prtol,max_iter=max_iter,**options)


@register('cyroot.scalar.bracketing')
@cython.binding(True)
def chandrupatla(f,a,b,f_a=None,f_b=None,*, sigma=1e-12,
           etol=1e-10,ertol=8.881784197001252e-16,
           ptol=1e-12,prtol=8.881784197001252e-16,max_iter=200,**options):
    """Run native chandrupatla; use cyroot.native.BracketSolver for stepping/history."""
    from .native.bracketing import chandrupatla as solve
    return solve(f,a,b,f_a=f_a,f_b=f_b,sigma=sigma,
                 etol=etol,ertol=ertol,ptol=ptol,prtol=prtol,max_iter=max_iter,**options)


@register('cyroot.scalar.bracketing')
@cython.binding(True)
def toms748(f,a,b,f_a=None,f_b=None,*, k=2,mu=.5,
           etol=1e-10,ertol=8.881784197001252e-16,
           ptol=1e-12,prtol=8.881784197001252e-16,max_iter=200,**options):
    """Run native toms748; use cyroot.native.BracketSolver for stepping/history."""
    from .native.bracketing import toms748 as solve
    return solve(f,a,b,f_a=f_a,f_b=f_b,k=k,mu=mu,
                 etol=etol,ertol=ertol,ptol=ptol,prtol=prtol,max_iter=max_iter,**options)


@register('cyroot.scalar.bracketing')
@cython.binding(True)
def itp(f,a,b,f_a=None,f_b=None,*, k1=None,k2=2.,n0=1,
           etol=1e-10,ertol=8.881784197001252e-16,
           ptol=1e-12,prtol=8.881784197001252e-16,max_iter=200,**options):
    """Run native itp; use cyroot.native.BracketSolver for stepping/history."""
    from .native.bracketing import itp as solve
    return solve(f,a,b,f_a=f_a,f_b=f_b,k1=k1,k2=k2,n0=n0,
                 etol=etol,ertol=ertol,ptol=ptol,prtol=prtol,max_iter=max_iter,**options)


