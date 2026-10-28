"""Compiled native secant example with an owned parameter context."""
from cyroot.native.secant import Secant
from cyroot.native.scalar cimport make_scalar_callback


cdef struct Parameters:
    double target


cdef int residual(double x, double* out, void* context) noexcept nogil:
    out[0] = x * x - (<Parameters*>context).target
    return 0


cdef class Quadratic:
    cdef Parameters parameters

    def __init__(self, double target):
        self.parameters.target = target

    def callback(self):
        return make_scalar_callback(residual, <void*>&self.parameters, self)


def solve(double target=2., double x0=1., double x1=2.):
    parameters = Quadratic(target)
    solver = Secant(parameters.callback(), x0, x1, history=32,
                    etol=1e-12, ertol=0.)
    return solver.run(), solver.history
