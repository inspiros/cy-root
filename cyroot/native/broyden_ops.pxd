"""Trusted allocation-free Broyden and Klement secant updates."""
from libc.stddef cimport size_t

cdef extern from "broyden.hpp" namespace "cyroot::native" nogil:
    int broyden_update(const double* matrix, const double* direction,
                      const double* target, size_t n, double* out) noexcept
    int klement_update(const double* matrix, const double* direction,
                      const double* target, size_t n, double* out) noexcept
