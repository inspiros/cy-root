"""Trusted allocation-free column-scaled Powell dogleg helpers."""
from libc.stddef cimport size_t
cdef extern from "dogleg_math.hpp" namespace "cyroot::native" nogil:
    int dogleg_prepare(const double* f,const double* j,size_t n,size_t m,
        double* scales,double* a,double* rhs,double* g,double* scale,double* gradient) noexcept
    int dogleg_step(const double* a,const double* g,const double* gn,size_t n,
        size_t m,double radius,double* c,double* ag,double* out,int* kind) noexcept
