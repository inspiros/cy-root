"""Allocation-free runtime-order Householder step from raw derivatives.

The caller supplies disjoint valid spans and 2*(order+1) scratch entries.
Status 0 is success, 1 nonfinite input/arithmetic, 2 size error or breakdown.
Only the final ratio is converted to binary64; coefficient scratch is scaled.
"""
from libc.stddef cimport size_t
from libc.stdint cimport int64_t

cdef extern from "householder.hpp" namespace "cyroot::native" nogil:
    cdef cppclass ScaledCoefficient:
        double mantissa
        int64_t exponent

    int householder_step(const double* derivatives, size_t order,
                         ScaledCoefficient* workspace, size_t capacity,
                         double* out) noexcept
