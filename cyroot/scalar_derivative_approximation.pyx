# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from typing import Callable, Sequence, Optional, Union

import cython
from dynamic_default_args import dynamic_default_args, named_default

from ._defaults import FINITE_DIFF_STEP
from .fptr cimport DoubleScalarFPtr, PyDoubleScalarFPtr
from .ops.scalar_ops cimport binomial_coef
from .typing import VectorLike

__all__ = [
    'DerivativeApproximation',
    'finite_difference',
    'FiniteDifference',
]

################################################################################
# Base Class
################################################################################
# noinspection DuplicatedCode
cdef class DerivativeApproximation(DoubleScalarFPtr):
    def __init__(self, f: Union[DoubleScalarFPtr, Callable[[float], float]]):
        if isinstance(f, DoubleScalarFPtr):
            self.f = f
        else:
            self.f = PyDoubleScalarFPtr(f)

    cpdef double eval(self, double x) except *:
        raise NotImplementedError

    cpdef double eval_with_f_val(self, double x, double f_x) except *:
        raise NotImplementedError

################################################################################
# Finite Difference
################################################################################
# noinspection DuplicatedCode
cdef double finite_difference_kernel(
        DoubleScalarFPtr f,
        double x,
        double f_x,
        double h=1.,
        int order=1,
        int kind=0):
    cdef unsigned int i
    cdef double f_i, diff = 0.
    cdef int bin_coef, sgn = (-1) ** order if kind == 1 else 1
    for i in range(order + 1):
        bin_coef = binomial_coef(order, i)
        if kind == 1:  # forward
            f_i = f.eval(x + i * h) if i > 0 else f_x
            diff += sgn * bin_coef * f_i
        elif kind == -1:  # backward
            f_i = f.eval(x - i * h) if i > 0 else f_x
            diff += sgn * bin_coef * f_i
        else:  # central
            f_i = f.eval(x + (<double> order / 2 - i) * h) if 2 * i != order else f_x
            diff += sgn * bin_coef * f.eval(x + (order / 2 - i) * h)
        sgn = -sgn
    return diff / h ** order

def _finite_diference_args_check(h: Union[float, VectorLike], order: int, kind: int):
    if (isinstance(h, float) and h == 0) or \
            (isinstance(h, Sequence) and any(h[i] == 0 for i in range(len(h)))):
        raise ValueError('h must be non-zero.')
    if order < 1:
        raise ValueError('order must be positive number.')
    if kind not in [-1, 0, 1]:
        raise ValueError('kind must be either -1 (backward), 0 (central), '
                         f'or 1 (forward). Got {kind}.')

# noinspection DuplicatedCode
cdef class FiniteDifference(DerivativeApproximation):
    def __init__(self,
                 f: Union[DoubleScalarFPtr, Callable[[float], float]],
                 h: float = FINITE_DIFF_STEP,
                 order: int = 1,
                 kind: int = 0):
        super().__init__(f)
        # check args
        _finite_diference_args_check(h, order, kind)
        self.h = h
        self.order = order
        self.kind = kind

    cpdef double eval(self, double x) except *:
        self.n_f_calls += 1
        cdef double f_x = self.f.eval(x)
        return finite_difference_kernel(
            <DoubleScalarFPtr> self.f, x, f_x, self.h, self.order, self.kind)

    cpdef double eval_with_f_val(self, double x, double f_x) except *:
        self.n_f_calls += 1
        return finite_difference_kernel(
            <DoubleScalarFPtr> self.f, x, f_x, self.h, self.order, self.kind)

# noinspection DuplicatedCode
@dynamic_default_args()
@cython.binding(True)
def finite_difference(f: Callable[[float], float],
                      x: float,
                      f_x: Optional[float] = None,
                      h: float = named_default(FINITE_DIFF_STEP=FINITE_DIFF_STEP),
                      order: int = 1,
                      kind: int = 0):
    """
    Finite difference method.

    Args:
        f (function): Function for which the derivative is sought.
        x (float): Point at which the derivative is sought.
        f_x (float, optional): Value evaluated at point ``x``.
        h (float): Finite difference step. Defaults to {h}.
        order (int): Order of derivative to be estimated.
         Defaults to 1.
        kind (int): Type of finite difference, including ``1``
         for forward, ``-1`` for backward, and ``0`` for central.
         Defaults to 0.

    Returns:
        diff: Estimated derivative.
    """
    # check args
    _finite_diference_args_check(h, order, kind)

    f_wrapper = PyDoubleScalarFPtr.from_f(f)
    if f_x is None:
        f_x = f_wrapper(x)
    return finite_difference_kernel(<DoubleScalarFPtr> f_wrapper, x, f_x, h, order, kind)
