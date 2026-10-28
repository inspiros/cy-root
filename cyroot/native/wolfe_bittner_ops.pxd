"""Trusted allocation-free Wolfe-Bittner affine candidate construction."""
from libc.stddef cimport size_t

cdef extern from "wolfe_bittner.hpp" namespace "cyroot::native" nogil:
    int wolfe_bittner_candidate(const double* matrix, const double* weights,
                               const double* points, size_t rows, size_t k,
                               size_t n, double* candidate,
                               double* affine_error) noexcept
