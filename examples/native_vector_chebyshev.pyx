"""Native residual, Jacobian and Hessian for a coupled Chebyshev solve."""
from libc.stddef cimport size_t
from cyroot.native import VectorChebyshev
from cyroot.native.vector cimport make_vector_callback
from cyroot.native.tensor_callbacks cimport make_hessian_callback


cdef struct Parameters:
    double first_target
    double second_target


cdef int residual(const double* x,size_t n,double* out,void* context) noexcept nogil:
    if n != 2:
        return 1
    cdef Parameters* p = <Parameters*>context
    out[0]=x[0]*x[0]+x[0]*x[1]+x[1]-p.first_target
    out[1]=2*x[0]+x[1]*x[1]-p.second_target
    return 0


cdef int jacobian(const double* x,size_t n,double* out,void* context) noexcept nogil:
    if n != 2:
        return 2
    out[0],out[1],out[2],out[3]=2*x[0]+x[1],2.,x[0]+1.,2*x[1]
    return 0


cdef int hessian(const double* x,size_t n,double* out,void* context) noexcept nogil:
    if n != 2:
        return 3
    # Raw offset: residual+n*(input1+n*input2).
    out[0],out[1],out[2],out[3]=2.,0.,1.,0.
    out[4],out[5],out[6],out[7]=1.,0.,0.,2.
    return 0


cdef class CoupledSystem:
    cdef Parameters parameters

    def __init__(self):
        self.parameters.first_target=5.
        self.parameters.second_target=6.

    def callbacks(self):
        return (make_vector_callback(residual,2,<void*>&self.parameters,self),
                make_vector_callback(jacobian,2,<void*>&self.parameters,self,jacobian=True),
                make_hessian_callback(hessian,2,<void*>&self.parameters,self))


def solve(initial=(1.25,1.75),numerical=False):
    """Return the root near (1,2) and copied immutable iteration history."""
    owner=CoupledSystem()
    function,jac,hess=owner.callbacks()
    solver=VectorChebyshev(function,None if numerical else jac,None if numerical else hess,
                          initial,history=16,etol=1e-12,ertol=0.)
    return solver.run(),solver.history
