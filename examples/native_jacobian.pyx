"""A native residual with a native finite-difference Jacobian plan."""
from libc.stddef cimport size_t
from cyroot.native import JacobianPlan, VectorNewton
from cyroot.native.vector cimport make_vector_callback


cdef struct Parameters:
    double first_target
    double second_target


cdef int residual(const double* x, size_t n, double* out, void* context) noexcept nogil:
    if n != 2:
        return 1
    cdef Parameters* parameters = <Parameters*>context
    out[0] = x[0] * x[0] + x[1] - parameters.first_target
    out[1] = 2 * x[0] + x[1] * x[1] - parameters.second_target
    return 0


cdef class CoupledSystem:
    cdef Parameters parameters

    def __init__(self):
        self.parameters.first_target = 3.
        self.parameters.second_target = 6.

    def callback(self):
        return make_vector_callback(residual, 2, <void*>&self.parameters, self)


def solve(initial=(1.4, 1.7), scheme='central', step=None):
    """Return the (1, 2) root and copied iteration history without analytic J."""
    parameters = CoupledSystem()
    plan = JacobianPlan(2, scheme=scheme, step=step)
    solver = VectorNewton(parameters.callback(), plan, initial, history=32,
                          etol=1e-12, ertol=0.)
    return solver.run(), solver.history
