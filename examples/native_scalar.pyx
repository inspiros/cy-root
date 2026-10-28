"""Compile with native_scalar_setup.py; call solve(target=2.) from Python."""
from cyroot.native.scalar import Bisection, Newton
from cyroot.native.scalar cimport make_scalar_callback


cdef struct Parameters:
    double target


cdef int residual(double x, double* out, void* context) noexcept nogil:
    cdef Parameters* parameters = <Parameters*>context
    out[0] = x * x - parameters.target
    return 0


cdef int derivative(double x, double* out, void* context) noexcept nogil:
    out[0] = 2 * x
    return 0


cdef class Quadratic:
    cdef Parameters parameters

    def __init__(self, double target):
        self.parameters.target = target

    def callback(self):
        # The descriptor retains self, so its native context stays alive.
        return make_scalar_callback(residual, <void*>&self.parameters, self)

    def derivative_callback(self):
        return make_scalar_callback(derivative, <void*>&self.parameters, self)


def solve(double target=2., double lower=0., double upper=2.):
    parameters = Quadratic(target)
    solver = Bisection(parameters.callback(), lower, upper, history=64)
    result = solver.run()
    return result, solver.history


def solve_newton(double target=2., double initial=1.5):
    """Solve with analytic native residual and derivative callbacks."""
    parameters = Quadratic(target)
    solver = Newton(parameters.callback(), parameters.derivative_callback(),
                    initial, history=64)
    result = solver.run()
    return result, solver.history
