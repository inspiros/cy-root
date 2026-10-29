"""Native resumable inverse-BFGS state, using an existing objective adapter."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from .buffers cimport OwnedDoubleBuffer
from .objective cimport LeastSquaresObjective
from .derivatives cimport RectangularJacobianPlan

cdef struct BFGSRecord:
    double error,precision,value,gradient,step_size
    size_t iters,f_calls,jac_calls,rejected_trials,skipped_updates,restarts
    int reason,callback_code,callback_stage

cdef class BFGS:
    cdef LeastSquaresObjective _objective
    cdef RectangularJacobianPlan _difference
    cdef OwnedDoubleBuffer *_storage,*_history_data
    cdef double *_x,*_f,*_g,*_trial,*_trial_f,*_trial_g,*_j,*_direction,*_dx
    cdef double *_h,*_candidate,*_u,*_v,*_w,*_diff_work
    cdef double *_pairs_u,*_pairs_v,*_ratios,*_curvatures,*_alphas
    cdef size_t _corrections,_pair_count,_pair_next
    cdef double _pair_scale
    cdef bint _limited
    cdef size_t _n,_m,_square,_memory_bytes,_record_count,_history_every
    cdef size_t _max_iter,_max_evaluations,_max_line_search
    cdef double _scale,_etol,_ertol,_ptol,_prtol,_gtol,_c1,_c2
    cdef vector[BFGSRecord] _records
    cdef BFGSRecord _state
    cdef bint _initialized,_constructing,_busy,_history_truncated
    cdef void _setup(self,object objective,object x0,object etol,object ertol,
        object ptol,object prtol,object gtol,object max_iter,object max_evaluations,
        object max_line_search,object c1,object c2,object history,object history_every,
        object workspace_bytes,object corrections) except *
    cdef void _reset_model(self) noexcept nogil
    cdef int _direction_model(self,double* slope) noexcept nogil
    cdef int _update_model(self) noexcept nogil
    cdef void _ready(self) except *
    cdef void _enter(self) except *
    cdef void _leave(self) noexcept
    cdef int _evaluate(self,const double* point,double* out) noexcept nogil
    cdef int _difference_sample(self,const double* point,double* out) noexcept nogil
    cdef int _gradient(self,const double* point,const double* f,double* out) noexcept nogil
    cdef bint _stationary(self) noexcept nogil
    cdef void _fail(self,int status) noexcept nogil
    cdef void _record(self) noexcept nogil
    # Trusted callers pin state/owners and hold _enter across native execution
    # and exports, then _leave with GIL. Numerical loops allocate no workspace;
    # Python providers acquire GIL. Terminated states are no-ops.
    cdef void _step(self) noexcept nogil
    cdef void _run(self) noexcept nogil
    cdef void _raise_callback_error(self) except *
    cdef object _snapshot(self,BFGSRecord record,const double* x,const double* f)


cdef class LBFGS(BFGS):
    pass
