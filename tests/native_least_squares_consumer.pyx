"""External installed-package consumer of the allocation-free DGELSY plan."""
from libc.stddef cimport size_t
from libc.math cimport NAN
from libcpp.vector cimport vector
from cyroot.native.buffers cimport OwnedDoubleBuffer
from cyroot.native.least_squares cimport LeastSquaresPlan

cdef extern from 'Python.h':
    int PyGILState_Check() noexcept nogil


def compute(LeastSquaresPlan plan, matrix, rhs, *, repeat=1, null=None):
    """Poison all reusable work, surround output with untouched sentinels."""
    cdef size_t m=len(matrix), n=len(matrix[0]), i, j, k, count=repeat
    cdef size_t work_count=plan._work_size if plan._work_size else 1
    cdef OwnedDoubleBuffer a=OwnedDoubleBuffer(m*n)
    cdef OwnedDoubleBuffer b=OwnedDoubleBuffer(m)
    cdef OwnedDoubleBuffer out=OwnedDoubleBuffer(n+2)
    cdef OwnedDoubleBuffer scratch=OwnedDoubleBuffer(work_count)
    cdef vector[int] pivots
    cdef int rank=111, info=222, status=-1, held=0
    cdef const double* ap=a.data()
    cdef const double* bp=b.data()
    cdef double* op=out.data()+1
    cdef double* wp=scratch.data()
    cdef int* pp
    cdef int* rp=&rank
    cdef int* ip=&info
    pivots.resize(n)
    pp=pivots.data()
    for i in range(m):
        b.data()[i]=rhs[i]
        for j in range(n):a.data()[i+m*j]=matrix[i][j]
    out.fill(987654.)
    scratch.fill(NAN)
    for i in range(n):pivots[i]=1  # Must not accidentally constrain pivoting.
    if null=='matrix':ap=NULL
    elif null=='rhs':bp=NULL
    elif null=='output':op=NULL
    elif null=='scratch':wp=NULL
    elif null=='pivots':pp=NULL
    elif null=='rank':rp=NULL
    elif null=='info':ip=NULL
    with nogil:
        held=PyGILState_Check()
        for k in range(count):
            status=plan.compute(ap,bp,op,wp,pp,rp,ip)
            if status!=0:break
    return (status,rank,info,tuple(out.data()[1+i] for i in range(n)),
            (out.data()[0],out.data()[n+1]),held,
            tuple(tuple(a.data()[i+m*j] for j in range(n)) for i in range(m)),
            tuple(b.data()[i] for i in range(m)))


def reuse(LeastSquaresPlan plan, matrix, right_hand_sides):
    """Reuse dirty factor/RHS/pivot allocations for distinct right hand sides."""
    cdef size_t m=len(matrix),n=len(matrix[0]),i,j
    cdef OwnedDoubleBuffer a=OwnedDoubleBuffer(m*n)
    cdef OwnedDoubleBuffer b=OwnedDoubleBuffer(m)
    cdef OwnedDoubleBuffer out=OwnedDoubleBuffer(n)
    cdef OwnedDoubleBuffer work=OwnedDoubleBuffer(plan.work_size)
    cdef vector[int] pivots
    cdef int rank,info,status
    pivots.resize(n)
    result=[]
    for i in range(m):
        for j in range(n):a.data()[i+m*j]=matrix[i][j]
    work.fill(NAN)
    for i in range(n):pivots[i]=1
    for values in right_hand_sides:
        for i in range(m):b.data()[i]=values[i]
        with nogil:status=plan.compute(a.data(),b.data(),out.data(),work.data(),pivots.data(),&rank,&info)
        result.append((status,rank,info,tuple(out.data()[i] for i in range(n))))
    return result
