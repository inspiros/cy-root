"""Immutable square-residual Hessian planning and allocation-free native work."""
from libc.stddef cimport size_t
from .buffers cimport OwnedDoubleBuffer
from .derivatives cimport ResidualFunction
from .rectangular cimport RectangularFunction


cdef struct HessianConfig:
    size_t dimension
    size_t square
    size_t cube
    size_t work
    int scheme
    const double* steps


# Trusted disjoint spans and nonnull callback/counter pointers. The config and
# borrowed steps must remain alive and immutable. Capacities are in doubles.
# Invalid configuration/capacities change only callback_code, resetting it to0.
# Valid spans: all failures poison out and restore the trial point to x.
# Workspace order (all segments double): axis cache2*n*n, first coordinates n,
# second coordinates n, first normalized deltas n, second normalized deltas n,
# scales n, alpha_first n, alpha_second n, trial point n, corner n, compensation n.
# Axis cache offset=(2*axis+side)*n+component, side0first/side1second.
# Trial point starts at 2*n*n+7*n. Output offset=o+n*(j+n*k).
cdef int compute_hessian(const HessianConfig* config,
                         const double* x, const double* f0,
                         double* out, size_t output_capacity,
                         double* workspace, size_t capacity,
                         ResidualFunction function, void* context,
                         size_t* calls, int* callback_code) noexcept nogil


cdef class HessianPlan:
    cdef size_t _dimension, _square, _cube, _work, _evaluation_size, _step_count
    cdef size_t _memory_bytes
    cdef int _scheme
    cdef OwnedDoubleBuffer* _steps
    cdef bint _initialized, _constructing
    cdef void copy_config(self, HessianConfig* config) noexcept nogil
    cdef int compute(self, const double* x, const double* f0,
                     double* out, size_t output_capacity,
                     double* workspace, size_t capacity,
                     ResidualFunction function, void* context,
                     size_t* calls, int* callback_code) noexcept nogil


cdef struct RectangularHessianConfig:
    size_t input_dimension
    size_t output_dimension
    size_t matrix_size
    size_t tensor_size
    size_t work
    int scheme
    const double* steps


# Trusted disjoint x[n],base[m],out[m*n*n],work[2*n*m+8*n+2*m].
# Config, borrowed steps and context stay live/immutable; pointers non-NULL.
# Axis cache offset (2*axis+side)*m+component; trial at 2*n*m+7*n.
# Tensor offset o+m*(j+n*k). Invalid config/capacities touch only callback_code.
# Valid-span failures poison the whole tensor and restore trial to x.
cdef int compute_rectangular_hessian(const RectangularHessianConfig* config,
    const double* x, const double* f0, double* out, size_t output_capacity,
    double* workspace, size_t capacity, RectangularFunction function,
    void* context, size_t* calls, int* callback_code) noexcept nogil

cdef class RectangularHessianPlan:
    cdef size_t _n, _m, _matrix_size, _tensor_size, _work, _evaluation_size
    cdef size_t _step_count, _memory_bytes
    cdef int _scheme
    cdef OwnedDoubleBuffer* _steps
    cdef bint _initialized, _constructing
    cdef void copy_config(self, RectangularHessianConfig* config) noexcept nogil
    cdef int compute(self, const double* x, const double* f0,
                     double* out, size_t output_capacity,
                     double* workspace, size_t capacity,
                     RectangularFunction function, void* context,
                     size_t* calls, int* callback_code) noexcept nogil

# Shared guarded boundary normalization, used by higher-order tensor plans.
cdef void _pack_vector(object source, double* out, size_t n) except *
cdef int _parse_scheme(object scheme) except -1
cdef OwnedDoubleBuffer* _copy_steps(size_t n, object step) except? NULL
