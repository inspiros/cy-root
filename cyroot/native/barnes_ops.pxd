"""Trusted allocation-free Barnes projection and multisecant update."""
from libc.stddef cimport size_t

cdef extern from "barnes.hpp" namespace "cyroot::native" nogil:
    int barnes_direction(const double* history, size_t count,
                         const double* step, size_t n,
                         double* basis, double* row, double* z) noexcept
    int barnes_update(const double* matrix, const double* step,
                      const double* target, const double* z,
                      size_t n, double* out) noexcept
