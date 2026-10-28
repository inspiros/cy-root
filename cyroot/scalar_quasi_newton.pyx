# distutils: language=c++
# cython: language_level=3
"""Established scalar interpolation names backed by native state machines."""
import cython
from .utils._function_registering import register

__all__ = ['secant','sidi','steffensen','inverse_quadratic_interp','hyperbolic_interp','muller']

@register('cyroot.scalar.quasi_newton')
@cython.binding(True)
def secant(f,x0,x1,f_x0=None,f_x1=None,
           *, etol=1e-10,ertol=8.881784197001252e-16,
           ptol=1e-12,prtol=8.881784197001252e-16,max_iter=200,**options):
    """Run native secant; native solver classes expose steps and copied history."""
    from .native.secant import secant as solve
    return solve(f,x0,x1,f_x0=f_x0,f_x1=f_x1,
                 etol=etol,ertol=ertol,ptol=ptol,prtol=prtol,max_iter=max_iter,**options)


@register('cyroot.scalar.quasi_newton')
@cython.binding(True)
def sidi(f,x0s,f_x0s=None,
           *, etol=1e-10,ertol=8.881784197001252e-16,
           ptol=1e-12,prtol=8.881784197001252e-16,max_iter=200,**options):
    """Run native sidi; native solver classes expose steps and copied history."""
    from .native.interpolation import sidi as solve
    return solve(f,x0s,f_x0s=f_x0s,
                 etol=etol,ertol=ertol,ptol=ptol,prtol=prtol,max_iter=max_iter,**options)


@register('cyroot.scalar.quasi_newton')
@cython.binding(True)
def steffensen(f,x0,f_x0=None,*, aitken=True,
           etol=1e-10,ertol=8.881784197001252e-16,
           ptol=1e-12,prtol=8.881784197001252e-16,max_iter=200,**options):
    """Run native steffensen; native solver classes expose steps and copied history."""
    from .native.interpolation import steffensen as solve
    return solve(f,x0,f_x0=f_x0,aitken=aitken,
                 etol=etol,ertol=ertol,ptol=ptol,prtol=prtol,max_iter=max_iter,**options)


@register('cyroot.scalar.quasi_newton')
@cython.binding(True)
def inverse_quadratic_interp(f,x0,x1,x2,f_x0=None,f_x1=None,f_x2=None,
           *, etol=1e-10,ertol=8.881784197001252e-16,
           ptol=1e-12,prtol=8.881784197001252e-16,max_iter=200,**options):
    """Run native inverse_quadratic_interp; native solver classes expose steps and copied history."""
    from .native.interpolation import inverse_quadratic_interp as solve
    return solve(f,x0,x1,x2,f_x0=f_x0,f_x1=f_x1,f_x2=f_x2,
                 etol=etol,ertol=ertol,ptol=ptol,prtol=prtol,max_iter=max_iter,**options)


@register('cyroot.scalar.quasi_newton')
@cython.binding(True)
def hyperbolic_interp(f,x0,x1,x2,f_x0=None,f_x1=None,f_x2=None,
           *, etol=1e-10,ertol=8.881784197001252e-16,
           ptol=1e-12,prtol=8.881784197001252e-16,max_iter=200,**options):
    """Run native hyperbolic_interp; native solver classes expose steps and copied history."""
    from .native.interpolation import hyperbolic_interp as solve
    return solve(f,x0,x1,x2,f_x0=f_x0,f_x1=f_x1,f_x2=f_x2,
                 etol=etol,ertol=ertol,ptol=ptol,prtol=prtol,max_iter=max_iter,**options)


@register('cyroot.scalar.quasi_newton')
@cython.binding(True)
def muller(f,x0,x1,x2,f_x0=None,f_x1=None,f_x2=None,
           *, etol=1e-10,ertol=8.881784197001252e-16,
           ptol=1e-12,prtol=8.881784197001252e-16,max_iter=200,**options):
    """Run native muller; native solver classes expose steps and copied history."""
    from .native.interpolation import muller as solve
    return solve(f,x0,x1,x2,f_x0=f_x0,f_x1=f_x1,f_x2=f_x2,
                 etol=etol,ertol=ertol,ptol=ptol,prtol=prtol,max_iter=max_iter,**options)


