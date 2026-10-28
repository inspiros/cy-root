"""Runtime-order native Householder on a quadratic without formula tables."""
from cyroot.native import Householder
from cyroot.native.scalar cimport make_scalar_callback


cdef struct Parameters:
    double target


cdef int residual(double x, double* out, void* context) noexcept nogil:
    out[0] = x*x-(<Parameters*>context).target
    return 0


cdef int first(double x, double* out, void* context) noexcept nogil:
    out[0] = 2*x
    return 0


cdef int second(double x, double* out, void* context) noexcept nogil:
    out[0] = 2.
    return 0


cdef int higher(double x, double* out, void* context) noexcept nogil:
    out[0] = 0.
    return 0


cdef class Quadratic:
    cdef Parameters parameters

    def __init__(self, double target):
        self.parameters.target = target

    def function(self):
        return make_scalar_callback(residual, <void*>&self.parameters, self)

    def derivatives(self, order):
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError('order must be a positive integer')
        first_callback = make_scalar_callback(first, <void*>&self.parameters, self)
        second_callback = make_scalar_callback(second, <void*>&self.parameters, self)
        higher_callback = make_scalar_callback(higher, <void*>&self.parameters, self)
        return tuple(first_callback if k == 1 else second_callback if k == 2 else higher_callback
                     for k in range(1, order+1))


def solve(order=3, double target=2., double initial=1.5):
    parameters = Quadratic(target)
    solver = Householder(parameters.function(), parameters.derivatives(order), initial,
                         history=16, etol=1e-12, ertol=0.)
    return solver.run(), solver.history
