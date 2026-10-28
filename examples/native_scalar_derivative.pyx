"""A native scalar residual with an automatic finite-difference derivative."""
from cyroot.native import Newton, ScalarDerivativePlan
from cyroot.native.scalar cimport make_scalar_callback


cdef struct Parameters:
    double target


cdef int residual(double x, double* out, void* context) noexcept nogil:
    cdef Parameters* parameters = <Parameters*>context
    out[0] = x * x - parameters.target
    return 0


cdef class Quadratic:
    cdef Parameters parameters

    def __init__(self, target=2.):
        self.parameters.target = target

    def callback(self):
        return make_scalar_callback(residual, <void*>&self.parameters, self)


def solve(initial=1.5, scheme='central', step=None):
    """Return the square root of two and copied iteration history."""
    owner = Quadratic()
    plan = ScalarDerivativePlan(scheme=scheme, step=step)
    solver = Newton(owner.callback(), plan, initial, history=32,
                    etol=1e-12, ertol=0.)
    return solver.run(), solver.history
