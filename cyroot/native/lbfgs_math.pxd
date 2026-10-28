"""Trusted normalized L-BFGS ring helpers; sized disjoint spans, capacity>0."""
from libc.stddef cimport size_t
cdef extern from "lbfgs_math.hpp" namespace "cyroot::native" nogil:
    int lbfgs_direction(const double* g,size_t n,size_t capacity,size_t count,
        size_t next,const double* us,const double* vs,const double* ratios,
        const double* curvatures,double gamma,double* alphas,double* q,
        double* direction,double* slope) noexcept
    int lbfgs_store(const double* g,const double* next_g,const double* step,
        size_t n,size_t capacity,size_t* count,size_t* next,double* us,double* vs,
        double* ratios,double* curvatures,double* gamma,double* u,double* v) noexcept
