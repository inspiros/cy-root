"""External low-order objective consumer with native callbacks and guarded raw calls."""
from libc.stddef cimport size_t
from libc.math cimport NAN
from libcpp.vector cimport vector
from cyroot.native.objective cimport LeastSquaresObjective
from cyroot.native.rectangular cimport make_rectangular_callback
from cyroot.native.vector cimport make_vector_callback
from cyroot.native.tensor_callbacks cimport (make_hessian_callback,
                                              make_rectangular_hessian_callback)

cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil

cdef struct Parameters:
    size_t calls[3]
    size_t gil_held
    int fail_stage
    int partial_stage

cdef int evaluate(const double* x, size_t n, size_t m, double* out,
                  void* context, int stage) noexcept nogil:
    cdef Parameters* p = <Parameters*>context
    cdef size_t i, count
    p.calls[stage-1] += 1
    if PyGILState_Check():p.gil_held += 1
    if p.fail_stage == stage:return -23-stage
    if n != 2:return -99
    count = m if stage == 1 else 2*m if stage == 2 else 4*m
    for i in range(count):out[i] = 0.
    for i in range(m):
        if stage == 1:out[i] = (i+1)*(x[0]*x[0]+3*x[1]-1)+i
        elif stage == 2:
            out[i] = (i+1)*2*x[0]
            out[i+m] = (i+1)*3
        else:out[i] = (i+1)*2
    if p.partial_stage == stage:
        # Simulate code0 with incomplete output, after explicitly restoring NaN.
        out[count-1] = NAN
    return 0


cdef int residual(const double* x, size_t n, size_t m, double* out, void* context) noexcept nogil:
    return evaluate(x,n,m,out,context,1)
cdef int jacobian(const double* x, size_t n, size_t m, double* out, void* context) noexcept nogil:
    return evaluate(x,n,m,out,context,2)
cdef int hessian(const double* x, size_t n, size_t m, double* out, void* context) noexcept nogil:
    return evaluate(x,n,m,out,context,3)
cdef int square_residual(const double* x, size_t n, double* out, void* context) noexcept nogil:
    return evaluate(x,n,n,out,context,1)
cdef int square_jacobian(const double* x, size_t n, double* out, void* context) noexcept nogil:
    return evaluate(x,n,n,out,context,2)
cdef int square_hessian(const double* x, size_t n, double* out, void* context) noexcept nogil:
    return evaluate(x,n,n,out,context,3)

cdef class Owner:
    cdef Parameters params
    cdef size_t m
    cdef object __weakref__
    def __init__(self,m=3,fail_stage=0,partial_stage=0):
        self.m=m
        self.params.fail_stage=fail_stage
        self.params.partial_stage=partial_stage
    def callbacks(self,square=False):
        if square:
            if self.m != 2:raise ValueError('Square fixture requires m=2.')
            return (make_vector_callback(square_residual,2,<void*>&self.params,self),
                    make_vector_callback(square_jacobian,2,<void*>&self.params,self,True),
                    make_hessian_callback(square_hessian,2,<void*>&self.params,self))
        return (make_rectangular_callback(residual,2,self.m,<void*>&self.params,self),
                make_rectangular_callback(jacobian,2,self.m,<void*>&self.params,self,True),
                make_rectangular_hessian_callback(hessian,2,self.m,<void*>&self.params,self))
    @property
    def calls(self):return tuple(self.params.calls[i] for i in range(3))
    @property
    def gil_held(self):return self.params.gil_held


def raw(LeastSquaresObjective objective, int order, point=(.5,.25), *,
        capacity=None, reserve=True, repeats=1):
    cdef vector[double] x,out
    cdef size_t needed, supplied, i, repeat, repeat_count = repeats
    cdef int status=-1, code=0, stage=0
    if len(point) != objective._n:raise ValueError('Point dimension mismatch.')
    x.resize(objective._n)
    for i in range(objective._n):x[i]=point[i]
    needed=objective._m if order==-1 else 1 if order==0 else objective._n if order==1 else objective._n*objective._n
    supplied=needed if capacity is None else capacity
    out.resize(max(needed,supplied)+2,1234.)
    if reserve:objective._enter()
    try:
        with nogil:
            for repeat in range(repeat_count):
                status=objective.evaluate_raw(x.data(),order,out.data()+1,supplied)
        code,stage=objective._callback_code,objective._callback_stage
    finally:
        if reserve:objective._leave()
    return (status,tuple(out[i+1] for i in range(needed)),code,stage,
            (out[0],out[out.size()-1]))


def shared_callback_reserved(LeastSquaresObjective objective, callback, point=(.5,.25)):
    """Hold objective's reservation while another public callback call is attempted."""
    other = LeastSquaresObjective(callback, objective._n, objective._m)
    objective._enter()
    try:return other.value(point)
    finally:objective._leave()
