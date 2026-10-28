"""Bounded sign-polygon discovery; caller owns storage and callback reservation."""
from libc.stddef cimport size_t
from .rectangular cimport RectangularFunction

cdef enum PolygonStatus:
    POLYGON_COMPLETE = 0
    POLYGON_INCOMPLETE = 1
    POLYGON_NONFINITE = 2
    POLYGON_CALLBACK_ERROR = 3
    POLYGON_BUDGET = 4

cdef size_t polygon_workspace_size(size_t n, size_t m) except? 0
cdef int polygon_sign_code(const double* values, size_t m, size_t* code) noexcept nogil
cdef bint polygon_cube_adjacent(size_t first, size_t second) noexcept nogil
cdef void polygon_fill_signs(double* signs, size_t m) noexcept nogil
cdef int construct_polygon(RectangularFunction function, void* context,
    const double* x, const double* initial_f, const double* h, size_t n, size_t m,
    double eps, size_t max_evaluations, size_t max_visits, size_t max_bisections,
    double* vertices, double* values, double* workspace,
    size_t* evaluations, size_t* visits, size_t* found, int* callback_code) noexcept nogil
