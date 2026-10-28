"""External scalar callback ABI and objective reservation fixture."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from cyroot.native.scalar cimport ScalarCallback,make_scalar_callback
from cyroot.native.objective cimport LeastSquaresObjective
from cyroot.native.bfgs cimport BFGS
from cyroot.native.levenberg_marquardt cimport LevenbergMarquardt

cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil

cdef struct Parameters:
    size_t calls[3]
    size_t gil_held
    int fail_stage

cdef int f(double x,double* out,void* context) noexcept nogil:
    cdef Parameters* p=<Parameters*>context
    p.calls[0]+=1
    if PyGILState_Check():p.gil_held+=1
    if p.fail_stage==1:return -61
    out[0]=x*x-2.
    return 0
cdef int j(double x,double* out,void* context) noexcept nogil:
    cdef Parameters* p=<Parameters*>context
    p.calls[1]+=1
    if PyGILState_Check():p.gil_held+=1
    if p.fail_stage==2:return -62
    out[0]=2*x
    return 0
cdef int h(double x,double* out,void* context) noexcept nogil:
    cdef Parameters* p=<Parameters*>context
    p.calls[2]+=1
    if PyGILState_Check():p.gil_held+=1
    if p.fail_stage==3:return -63
    out[0]=2.
    return 0

cdef class Owner:
    cdef Parameters params
    cdef object __weakref__
    def callbacks(self):
        return (make_scalar_callback(f,<void*>&self.params,self),
                make_scalar_callback(j,<void*>&self.params,self),
                make_scalar_callback(h,<void*>&self.params,self))
    @property
    def counters(self):return tuple(self.params.calls[i] for i in range(3)),self.params.gil_held
    @property
    def fail_stage(self):return self.params.fail_stage
    @fail_stage.setter
    def fail_stage(self,value):self.params.fail_stage=value


def active(ScalarCallback callback):return bool(callback._active)


def raw_objective(LeastSquaresObjective objective,int order,point=1.5):
    cdef double x=point,out=1234.
    cdef int status
    objective._enter()
    try:
        with nogil:status=objective.evaluate_raw(&x,order,&out,1)
        objective._raise_callback_error(out)
        return status,out
    finally:objective._leave()


def run_raw(state):
    cdef BFGS quasi
    cdef LevenbergMarquardt trust
    if isinstance(state,BFGS):
        quasi=state;quasi._enter()
        try:
            with nogil:quasi._run()
            quasi._raise_callback_error()
        finally:quasi._leave()
    else:
        trust=state;trust._enter()
        try:
            with nogil:trust._run()
            trust._raise_callback_error()
        finally:trust._leave()
    return state.result
