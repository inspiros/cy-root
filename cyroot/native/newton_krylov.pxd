"""Native Newton-Krylov state with bounded GMRES/LSQR storage."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from .buffers cimport OwnedDoubleBuffer
from .vector cimport VectorCallback
from .rectangular cimport RectangularCallback
from .jvp cimport JacobianVectorCallback,JacobianVectorPlan
from .vjp cimport JacobianTransposeCallback

cdef extern from "objective.hpp" namespace "cyroot::native" nogil:
    cdef cppclass ObjectiveAccumulator:
        ObjectiveAccumulator() noexcept
        void add_product(double,double) noexcept
        int value(double*) noexcept const
from cyroot._stopping cimport StopState

cdef struct KrylovRecord:
    double error,precision,linear_residual,linear_stationarity,step_scale
    size_t iters,f_calls,jvp_calls,vjp_calls,linear_iterations,backtracks
    int reason,linear_status,callback_code,callback_stage

cdef class NewtonKrylov:
    cdef VectorCallback _square_function
    cdef RectangularCallback _function
    cdef JacobianVectorCallback _product_callback
    cdef JacobianVectorPlan _plan
    cdef JacobianTransposeCallback _transpose_callback
    cdef vector[ObjectiveAccumulator] _sums
    cdef OwnedDoubleBuffer* _storage
    cdef OwnedDoubleBuffer* _history_data
    cdef double* _x
    cdef double* _f
    cdef double* _delta
    cdef double* _trial
    cdef double* _trial_f
    cdef double* _applied
    cdef double* _rhs
    cdef double* _difference_work
    cdef double* _gmres_work
    cdef double* _basis
    cdef double* _column
    cdef size_t _n,_m,_restart,_max_linear_iter,_max_iter,_max_evaluations,_max_line_search
    cdef size_t _record_count,_history_every,_memory_bytes
    cdef double _linear_rtol,_c1,_contraction,_scale
    cdef KrylovRecord _state
    cdef StopState _stop
    cdef vector[KrylovRecord] _records
    cdef bint _initialized,_constructing,_busy,_history_truncated,_numerical_pair
    cdef void _enter(self) except *
    cdef void _leave(self) noexcept
    cdef void _ready(self) except *
    cdef int _evaluate(self,const double* point,double* out) noexcept nogil
    cdef int _product(self,const double* direction,double* out) noexcept nogil
    cdef int _apply(self,const double* direction,double* out) noexcept nogil
    cdef int _transpose(self,const double* direction,double* out) noexcept nogil
    cdef int _columns(self,const double* direction,double* out,bint transpose) noexcept nogil
    cdef void _check(self) noexcept nogil
    cdef void _record(self) noexcept nogil
    cdef void _step(self) noexcept nogil
    cdef void _run(self) noexcept nogil
    cdef void _raise_callback_error(self) except *
    cdef object _snapshot(self,KrylovRecord record,const double* x,const double* f)
