"""Homeier shares Newton's callback, state and bounded-history contract."""
from .scalar cimport Newton

cdef class Homeier(Newton):
    cdef int _derivative_at(self, double x, double f_x, double* out) noexcept nogil
    cdef void _step(self) noexcept nogil
