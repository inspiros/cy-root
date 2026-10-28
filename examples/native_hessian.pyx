"""A native square residual with a standalone finite-difference Hessian."""
from libc.stddef cimport size_t
from cyroot.native import HessianPlan
from cyroot.native.vector cimport make_vector_callback


cdef struct Parameters:
    double coupling


cdef int residual(const double* x, size_t n, double* out, void* context) noexcept nogil:
    if n != 2:
        return 1
    cdef double coupling = (<Parameters*>context).coupling
    out[0] = x[0]*x[0]+coupling*x[0]*x[1]+2*x[1]*x[1]
    out[1] = 5*x[0]*x[0]-7*x[0]*x[1]+11*x[1]*x[1]
    return 0


cdef class QuadraticSystem:
    cdef Parameters parameters

    def __init__(self, double coupling=3.):
        self.parameters.coupling = coupling

    def callback(self):
        return make_vector_callback(residual, 2, <void*>&self.parameters, self)


def estimate(point=(.75, -1.25), scheme='central', step=None, coupling=3.):
    """Return copied tensor, residual-attempt count and status."""
    owner = QuadraticSystem(coupling)
    plan = HessianPlan(2, scheme=scheme, step=step)
    return plan.evaluate(owner.callback(), point)
