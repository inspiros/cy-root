# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""Vector bisection compatibility boundaries over the native Vrahatis engine."""
from cyroot.defaults cimport ETOL, ERTOL, PTOL, PRTOL, MAX_ITER
import cython
import numpy as np
from .utils._function_registering import register
from .native.scalar import _nonnegative_integer

__all__ = ['vrahatis']


@register('cyroot.vector.bracketing')
@cython.binding(True)
def vrahatis(F,x0s,F_x0s=None,*, h=None,etol=ETOL,ertol=ERTOL,
             ptol=PTOL,prtol=PRTOL,max_iter=MAX_ITER,**options):
    """Native sign-polygon bisection; use native.Vrahatis for step/history.

    Python residuals receive copied tuples. Explicit VectorCallback and
    RectangularCallback residuals permit GIL-free numerical execution. Vertex
    signs alone do not certify a multidimensional root; only residual tolerance
    gives root success when output dimension exceeds one.
    """
    from .native.vrahatis import Vrahatis
    return Vrahatis(F,x0s,F_x0s=F_x0s,h=h,etol=etol,ertol=ertol,
                    ptol=ptol,prtol=prtol,max_iter=max_iter,**options).run()


def compute_admissible_n_polygon(F,x,h=None,eps=1e-5,**options):
    """Construct a sign-complete polygon using bounded native sampling.

    ndarray conversion is only for the returned compatibility arrays. Failure
    to discover every strict output-sign pattern raises ValueError.
    """
    from .native.vrahatis import compute_admissible_n_polygon as construct
    points,signs = construct(F,x,h=h,eps=eps,**options)
    return np.asarray(points,dtype=np.float64),np.asarray(signs,dtype=np.float64)


def get_M(n,sign=False,*,workspace_bytes=64*1024*1024):
    """Binary/sign cube rows, with bounded output and checked bit counts."""
    cdef size_t dimension=_nonnegative_integer(n,'n'),count,i,j
    cdef size_t limit=_nonnegative_integer(workspace_bytes,'workspace_bytes')
    cdef bint signed=bool(sign)
    if dimension >= 8*sizeof(size_t):
        raise ValueError('Cube dimension exceeds the native bit capacity.')
    count=(<size_t>1)<<dimension
    if dimension and count > limit//sizeof(double)//dimension:
        raise ValueError('Cube rows exceed workspace_bytes.')
    result=np.empty((count,dimension),dtype=np.float64)
    cdef double[:,::1] output=result
    with nogil:
        for i in range(count):
            for j in range(dimension):
                output[i,j]=1. if (i >> (dimension-1-j)) & 1 else (-1. if signed else 0.)
    return result


def _get_1_simplexes(S,*,workspace_bytes=64*1024*1024):
    """Return unordered row pairs and their count of equal sign entries."""
    source=np.asarray(S,dtype=np.float64)
    if source.ndim != 2:raise ValueError('Signs must be a matrix.')
    cdef double[:,:] signs=source
    cdef size_t rows=source.shape[0],cols=source.shape[1],i,j,k,p=0,pairs
    limit=_nonnegative_integer(workspace_bytes,'workspace_bytes')
    # Python arithmetic validates products before size_t conversion/allocation.
    size=int(rows)*(int(rows)-1)//2
    if size > limit//(3*sizeof(size_t)):
        raise ValueError('Pair table exceeds workspace_bytes.')
    pairs=size
    result=np.zeros((pairs,3),dtype=np.uintp)
    cdef size_t[:,::1] output=result
    with nogil:
        for i in range(rows):
            for j in range(i+1,rows):
                output[p,0],output[p,1]=i,j
                for k in range(cols):output[p,2]+=signs[i,k]==signs[j,k]
                p+=1
    return result


def sorted_by_vertices(*mats,S):
    """Compatibility array ordering by lexicographic sign rows."""
    signs=np.asarray(S)
    if signs.ndim != 2:raise ValueError('Signs must be a matrix.')
    arrays=tuple(np.asarray(matrix) for matrix in mats)
    if any(array.ndim==0 or array.shape[0]!=signs.shape[0] for array in arrays):
        raise ValueError('Each matrix must have the same row count as signs.')
    order=sorted(range(signs.shape[0]),key=lambda i:tuple(signs[i]))
    return (*tuple(array[order] for array in arrays),signs[order])
