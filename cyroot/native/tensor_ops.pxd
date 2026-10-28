"""Allocation-free contractions of packed residual Hessians and matrices."""
from libc.stddef cimport size_t

cdef extern from "tensor_ops.hpp" namespace "cyroot::native" nogil:
    # Trusted disjoint valid H[n^3],direction[n],out[n] spans. Invalid checked
    # n returns2 without writes. Nonfinite input/final result returns1 and
    # poisons all output. Caller owns synchronization and pointer lifetimes.
    int contract_hessian(const double* hessian, const double* direction,
                         size_t n, double* out) noexcept
    # out[o+n*k]=sum_j H[o+n*(j+n*k)]*direction[j]. Trusted disjoint
    # H[n^3],direction[n],out[n^2]. Same status/preflight/poison guarantees.
    int contract_hessian_direction(const double* hessian, const double* direction,
                                   size_t n, double* out) noexcept
    double negative_half_sum(double a, double b) noexcept
    # Trusted disjoint column-major matrix[n^2],direction[n],out[n]. Same
    # statuses: invalid n2 before writes, nonfinite1 poisons entire output.
    int matrix_vector(const double* matrix, const double* direction,
                      size_t n, double* out) noexcept
    # H[o+m*(j+n*k)] with n inputs and m residuals; outputs have m and m*n
    # entries. All ordered pairs are used, with no Hessian symmetry assumption.
    # Invalid n/m or overflowing byte spans returns2 before writes; nonfinite
    # input/final result returns1 and poisons the complete output. Trusted,
    # disjoint spans and synchronization/lifetimes are the caller's responsibility.
    int contract_rectangular_hessian(const double* hessian,
                                     const double* direction, size_t n, size_t m,
                                     double* out) noexcept
    int contract_rectangular_hessian_direction(const double* hessian,
                                               const double* direction,
                                               size_t n, size_t m,
                                               double* out) noexcept
    # Column-major A[o+m*j], direction[n], out[m]; same guarantees.
    int rectangular_matrix_vector(const double* matrix, const double* direction,
                                  size_t n, size_t m, double* out) noexcept
