"""Compiled square vector Newton example with a non-symmetric Jacobian."""
from libc.stddef cimport size_t
from cyroot.native import VectorNewton
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


cdef int jacobian(const double* x, size_t n, double* out, void* context) noexcept nogil:
    if n != 2:
        return 1
    # Native matrices are column-major: row + column*n.
    out[0], out[1], out[2], out[3] = 2 * x[0], 2., 1., 2 * x[1]
    return 0


cdef class CoupledSystem:
    cdef Parameters parameters

    def __init__(self, double first_target=3., double second_target=6.):
        self.parameters.first_target = first_target
        self.parameters.second_target = second_target

    def function(self):
        return make_vector_callback(residual, 2, <void*>&self.parameters, self)

    def jacobian(self):
        return make_vector_callback(jacobian, 2, <void*>&self.parameters, self, jacobian=True)


def solve(initial=(1.4, 1.7)):
    """Return the (1, 2) root and copied native iteration snapshots."""
    parameters = CoupledSystem()
    solver = VectorNewton(parameters.function(), parameters.jacobian(), initial,
                          history=32, etol=1e-12, ertol=0.)
    return solver.run(), solver.history
