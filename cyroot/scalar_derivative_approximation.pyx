# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from typing import Callable, Optional, Union
from operator import index as integer_index
from math import isfinite

import cython
from libc cimport math

from ._types import VectorLike
from .fptr cimport DoubleScalarFPtr, PyDoubleScalarFPtr
from .utils._function_registering import register

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

    def __call__(self, double x) -> double:
        return self.eval(x)

    cpdef double eval_with_f_val(self, double x, double f_x):
        return self.eval2(x, f_x)

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
        int kind=0) nogil:
    # Keep validation here as the public attributes can be changed after construction.
    if order < 1 or kind < -1 or kind > 1:
        with gil:
            raise ValueError('order must be positive and kind must be -1, 0, or 1.')
    if not math.isfinite(h) or h == 0:
        with gil:
            raise ValueError('h must be finite and non-zero.')
    if not math.isfinite(x):
        with gil:
            raise ValueError('x must be finite.')
    cdef double scale = h ** order
    if not math.isfinite(scale) or scale == 0:
        with gil:
            raise ValueError('h ** order must be finite and non-zero.')
    cdef Py_ssize_t i
    cdef double f_i, sample_x, offset, previous_x = x, weighted, diff = 0.
    cdef double bin_coef = 1.
    cdef int sgn = -1 if kind == 1 and order % 2 else 1
    # Validate the complete one-dimensional stencil before taking its samples.
    # Distinct offsets can round to the same point even if neither equals x.
    for i in range(<Py_ssize_t> order + 1):
        if kind == 1:
            offset = i
        elif kind == -1:
            offset = -i
        else:
            offset = 0.5 * order - i
        sample_x = x + offset * h
        if (not math.isfinite(sample_x) or
                (offset != 0 and sample_x == x) or
                (i > 0 and sample_x == previous_x)):
            with gil:
                raise ValueError('h does not produce finite, distinct sample points.')
        previous_x = sample_x
        if i < order:
            bin_coef *= (<double> order - i) / (i + 1)
            if not math.isfinite(bin_coef):
                with gil:
                    raise ValueError('finite-difference binomial coefficients must be representable as finite doubles.')
    bin_coef = 1.
    for i in range(<Py_ssize_t> order + 1):
        if kind == 1:  # forward
            offset = i
        elif kind == -1:  # backward
            offset = -i
        else:  # central: division must be floating point for odd orders
            offset = 0.5 * order - i
        if offset == 0:
            f_i = f_x
        else:
            sample_x = x + offset * h
            if not math.isfinite(sample_x) or sample_x == x:
                with gil:
                    raise ValueError('h does not produce finite, distinct sample points.')
            f_i = f.eval(sample_x)
        if not math.isfinite(f_i):
            with gil:
                raise ValueError('f(x) and f_x must be finite.')
        weighted = sgn * bin_coef * f_i
        if not math.isfinite(weighted):
            with gil:
                raise ValueError('finite-difference weighted samples must be finite.')
        diff += weighted
        if not math.isfinite(diff):
            with gil:
                raise ValueError('finite-difference accumulators must be finite.')
        sgn = -sgn
        if i < order:
            # A floating recurrence avoids overflow in the old C-int binomial helper.
            bin_coef *= (<double> order - i) / (i + 1)
    diff /= scale
    if not math.isfinite(diff):
        with gil:
            raise ValueError('the finite-difference derivative must be finite.')
    return diff


def _check_finite_difference_args(h, order, kind):
    """Normalize Python inputs before dispatch to typed numerical kernels."""
    if isinstance(order, bool):
        raise ValueError('order must be a positive integer.')
    try:
        order = integer_index(order)
    except TypeError:
        raise ValueError('order must be a positive integer.') from None
    if order < 1 or order > 2147483647:
        raise ValueError('order must be a positive integer representable by a C int.')
    if isinstance(kind, str):
        kinds = {'forward': 1, 'backward': -1, 'central': 0}
        if kind not in kinds:
            raise ValueError("kind must be 1/'forward', -1/'backward', or 0/'central'.")
        kind = kinds[kind]
    else:
        try:
            kind = integer_index(kind)
        except TypeError:
            raise ValueError("kind must be 1/'forward', -1/'backward', or 0/'central'.") from None
        if kind not in (-1, 0, 1):
            raise ValueError("kind must be 1/'forward', -1/'backward', or 0/'central'.")
    if isinstance(h, (str, bytes)):
        raise ValueError('h must contain finite, non-zero real steps.')
    try:
        steps = iter(h)
    except TypeError:
        try:
            h = float(h)
        except (TypeError, ValueError, OverflowError):
            raise ValueError('h must contain finite, non-zero real steps.') from None
        if not isfinite(h) or h == 0:
            raise ValueError('h must be finite and non-zero.')
    else:
        try:
            h = tuple(float(step) for step in steps)
        except (TypeError, ValueError, OverflowError):
            raise ValueError('h must be a one-dimensional sequence of real steps.') from None
        if not h or any(not isfinite(step) or step == 0 for step in h):
            raise ValueError('h must contain finite, non-zero steps.')
    return h, order, kind

# noinspection DuplicatedCode
cdef class FiniteDifference(DerivativeApproximation):
    r"""A class to wrap a function for derivative approximation."""
    def __init__(self,
                 f: Union[DoubleScalarFPtr, Callable[[float], float]],
                 h: float = 1.,
                 order=1,
                 kind: Union[int, str] = 0):
        super().__init__(f)
        # check args
        h, order, kind = _check_finite_difference_args(h, order, kind)
        if not isinstance(h, float):
            raise ValueError('h must be a scalar step.')
        self.h = h
        self.order = order
        self.kind = kind

    cdef double eval(self, double x) except * nogil:
        if not math.isfinite(x):
            with gil:
                raise ValueError('x must be finite.')
        self.n_f_calls += 1
        cdef double f_x = 0.
        if self.kind != 0 or self.order % 2 == 0:
            f_x = self.f.eval(x)
        return finite_difference_kernel(
            <DoubleScalarFPtr> self.f, x, f_x, self.h, self.order, self.kind)

    cdef double eval2(self, double x, double f_x) except * nogil:
        self.n_f_calls += 1
        return finite_difference_kernel(
            <DoubleScalarFPtr> self.f, x, f_x, self.h, self.order, self.kind)

# noinspection DuplicatedCode
@register('cyroot.da.scalar')
@cython.binding(True)
def finite_difference(f: Callable[[float], float],
                      x: float,
                      f_x: Optional[float] = None,
                      h: float = 1.,
                      order=1,
                      kind: Union[int, str] = 0):
    r"""
    Finite difference method.

    Args:
        f (function): Function for which the derivative is sought.
        x (float): Point at which the derivative is sought.
        f_x (float, optional): Value evaluated at point ``x``.
        h (float): Finite difference step. Defaults to 1.
        order (int): Order of derivative to be estimated.
         Defaults to 1.
        kind (int, str): Type of finite difference, including ``1``
         for forward, ``-1`` for backward, and ``0`` for central.
         Defaults to 0.

    Returns:
        diff: Estimated derivative. Central differences use ``order + 1``
         equally spaced samples centered on ``x``. The base value is reused
         when it belongs to the stencil; odd central orders do not need it.
         Unrepresentable coefficients, collapsed sample points, and nonfinite
         weighted sums or derivatives raise ``ValueError``.
    """
    # check args
    h, order, kind = _check_finite_difference_args(h, order, kind)

    if not isfinite(x):
        raise ValueError('x must be finite.')
    f_wrapper = PyDoubleScalarFPtr.from_f(f)
    if f_x is None:
        f_x = f_wrapper.eval(x) if kind != 0 or order % 2 == 0 else 0.
    return finite_difference_kernel(f_wrapper, x, f_x, h, order, kind)
