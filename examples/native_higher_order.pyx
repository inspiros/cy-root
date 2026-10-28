"""Native Halley family with analytic or numerical derivatives."""
from cyroot.native import Halley, Chebyshev, ModifiedHalley, SuperHalley, TangentHyperbolas
from cyroot.native.scalar cimport make_scalar_callback


cdef struct Parameters:
    double target


cdef int residual(double x, double* out, void* context) noexcept nogil:
    out[0] = x*x - (<Parameters*>context).target
    return 0


cdef int derivative(double x, double* out, void* context) noexcept nogil:
    out[0] = 2*x
    return 0


cdef int second_derivative(double x, double* out, void* context) noexcept nogil:
    out[0] = 2.
    return 0


cdef class Quadratic:
    cdef Parameters parameters

    def __init__(self, double target):
        self.parameters.target = target

    def callbacks(self):
        return (make_scalar_callback(residual, <void*>&self.parameters, self),
                make_scalar_callback(derivative, <void*>&self.parameters, self),
                make_scalar_callback(second_derivative, <void*>&self.parameters, self))


def solve(method='halley', double target=2., double initial=1.5,
          derivatives='analytic', *, alpha=0.5):
    methods = {'halley': Halley, 'chebyshev': Chebyshev,
               'modified_halley': ModifiedHalley, 'super_halley': SuperHalley,
               'tangent_hyperbolas': TangentHyperbolas}
    if method not in methods:
        raise ValueError('Unknown Halley family method.')
    if derivatives not in ('analytic', 'numerical', 'mixed'):
        raise ValueError('derivatives must be analytic, numerical or mixed')
    owner = Quadratic(target)
    function, first, second = owner.callbacks()
    if derivatives == 'numerical':
        first = second = None
    elif derivatives == 'mixed':
        second = None
    solver_type = methods[method]
    options = {'alpha': alpha} if method == 'modified_halley' else {}
    solver = solver_type(function, first, second, initial, history=16,
                         etol=1e-12, ertol=0., **options)
    return solver.run(), solver.history
