"""Trusted allocation-free Robinson probe construction."""
from libc.stddef cimport size_t

cdef extern from "robinson.hpp" namespace "cyroot::native" nogil:
    int robinson_nodes(const double* previous, const double* current,
                       size_t n, bint identity, double* nodes,
                       double* transpose, double* scratch) noexcept
