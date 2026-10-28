"""Native finite differences and runtime-order Householder increment."""
from operator import index
from libc.math cimport exp
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from cyroot.native import ScalarFiniteDifferencePlan
from cyroot.native.scalar cimport make_scalar_callback
from cyroot.native.householder cimport ScaledCoefficient, householder_step


cdef int exponential(double x, double* out, void* context) noexcept nogil:
    out[0] = exp(x)
    return 0


def estimate_exponential(order=3, scheme='central', step=None):
    plan = ScalarFiniteDifferencePlan(order, scheme, step)
    return plan.evaluate(make_scalar_callback(exponential, NULL), .4)


def quadratic_householder_step(order=12, double point=1.5,
                               workspace_bytes=1024 * 1024):
    """One increment for x²-1; this example applies a configurable byte budget."""
    if isinstance(order, bool) or isinstance(workspace_bytes, bool):
        raise ValueError('order and workspace_bytes must be integers, not bool.')
    degree, budget = index(order), index(workspace_bytes)
    if degree < 1 or budget < 0:
        raise ValueError('order must be positive and workspace_bytes nonnegative.')
    required = (degree + 1) * (sizeof(double) + 2 * sizeof(ScaledCoefficient))
    if required > budget:
        raise ValueError('Example scratch exceeds workspace_bytes.')
    cdef size_t d = degree
    cdef vector[double] derivatives = vector[double](d + 1, 0.)
    cdef vector[ScaledCoefficient] scratch = vector[ScaledCoefficient](2 * (d + 1))
    cdef double increment
    cdef int status
    derivatives[0] = point * point - 1.
    derivatives[1] = 2. * point
    if d >= 2:
        derivatives[2] = 2.
    with nogil:
        status = householder_step(derivatives.data(), d, scratch.data(), scratch.size(), &increment)
    return status, increment, point + increment
