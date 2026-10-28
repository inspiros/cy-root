"""Compiled generalized bisection over a sign-complete square polygon."""
from libc.stddef cimport size_t
from cyroot.native import Vrahatis
from cyroot.native.vector cimport make_vector_callback

cdef int residual(const double* x,size_t n,double* out,void* context) noexcept nogil:
    if n != 2:return 1
    out[0],out[1] = x[0]-.7,x[1]-.7
    return 0


def callback():
    return make_vector_callback(residual,2,NULL,callback)


def solve():
    solver=Vrahatis(callback(),((0.,0.),(0.,2.),(2.,0.),(2.,2.)),history=64)
    return solver.run(),solver.history
