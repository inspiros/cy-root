"""Armijo-globalized Newton states; callback/storage contracts match Newton."""
from libc.stddef cimport size_t
from .scalar cimport Newton
from .vector cimport VectorNewton

cdef class DampedNewton(Newton):
    cdef object _line_options
    cdef double _c1, _contraction, _step_scale
    cdef size_t _max_line_search, _backtracks
    cdef void _enter(self) except *
    cdef void _step(self) noexcept nogil

cdef class VectorDampedNewton(VectorNewton):
    cdef object _line_options
    cdef double _c1, _contraction, _step_scale
    cdef size_t _max_line_search, _backtracks
    cdef void _enter(self) except *
    cdef void _step(self) noexcept nogil
