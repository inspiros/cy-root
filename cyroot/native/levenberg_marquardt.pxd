"""Resumable damped Gauss-Newton state using an existing objective adapter."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from .buffers cimport OwnedDoubleBuffer
from .objective cimport LeastSquaresObjective
from .least_squares cimport LeastSquaresPlan
from .derivatives cimport RectangularJacobianPlan

cdef struct LMRecord:
    double error, precision, damping, gradient, radius
    size_t iters, f_calls, jac_calls, rejected_trials
    int reason, linear_status, linear_rank, callback_code, callback_stage

cdef class LevenbergMarquardt:
    cdef LeastSquaresObjective _objective
    cdef LeastSquaresPlan _linear
    cdef RectangularJacobianPlan _difference
    cdef OwnedDoubleBuffer *_storage, *_history_data
    cdef double *_x, *_f, *_j, *_trial, *_trial_f, *_scales, *_z, *_dx
    cdef double *_augmented, *_rhs, *_work, *_diff_work
    cdef double *_dogleg_g,*_dogleg_c,*_dogleg_ag,*_dogleg_z
    cdef double _min_radius,_max_radius
    cdef bint _dogleg
    cdef size_t _n, _m, _rows, _memory_bytes, _record_count, _history_every
    cdef size_t _max_iter, _max_evaluations, _max_trials
    cdef double _initial_error, _etol, _ertol, _ptol, _prtol, _gtol
    cdef vector[int] _pivots
    cdef vector[LMRecord] _records
    cdef LMRecord _state
    cdef bint _initialized, _constructing, _busy, _history_truncated
    cdef void _setup(self,object objective,object x0,object etol,object ertol,
        object ptol,object prtol,object gtol,object max_iter,object max_evaluations,
        object max_trials,object damping,object rcond,object history,object history_every,
        object workspace_bytes,bint dogleg,object initial_radius,object max_radius,
        object min_radius) except *
    cdef void _step_dogleg(self) noexcept nogil
    cdef void _ready(self) except *
    cdef void _enter(self) except *
    cdef void _release(self) noexcept
    cdef int _evaluate(self, const double* point, double* out) noexcept nogil
    cdef int _difference_sample(self, const double* point, double* out) noexcept nogil
    cdef bint _stationary(self) noexcept nogil
    cdef void _fail(self, int status) noexcept nogil
    cdef void _record(self) noexcept nogil
    # Caller pins state and holds _enter reservation through native execution
    # and all result/error exports. No native allocation; Python callbacks use
    # the GIL. _step and _run are no-ops once terminated.
    cdef void _step(self) noexcept nogil
    cdef void _run(self) noexcept nogil
    cdef void _raise_callback_error(self) except *
    cdef object _snapshot(self, LMRecord record, const double* x, const double* f)


cdef class PowellDogleg(LevenbergMarquardt):
    pass
