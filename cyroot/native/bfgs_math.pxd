"""Allocation-free BFGS helpers; trusted sized/disjoint spans, column-major H."""
from libc.stddef cimport size_t
cdef extern from "bfgs_math.hpp" namespace "cyroot::native" nogil:
    int bfgs_value(const double* f,size_t m,double scale,double* out) noexcept
    int bfgs_gradient(const double* f,const double* j,size_t n,size_t m,double scale,double* out) noexcept
    int bfgs_stationarity(const double* f,const double* j,size_t n,size_t m,double* out) noexcept
    int bfgs_change(const double* f,const double* trial,size_t m,double scale,double* out) noexcept
    int bfgs_dot(const double* a,const double* b,size_t n,double* out) noexcept
    void bfgs_identity(double* h,size_t n) noexcept
    int bfgs_direction(const double* h,const double* g,size_t n,double* direction,double* slope) noexcept
    int bfgs_trial(const double* x,const double* p,size_t n,double alpha,double* trial,double* dx) noexcept
    int bfgs_update(double* h,const double* g,const double* next_g,const double* step,
                    size_t n,double* candidate,double* u,double* v,double* w) noexcept
