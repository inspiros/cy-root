"""Small external callback fixture for LM's compiled/no-GIL boundary."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from cyroot.native.rectangular cimport make_rectangular_callback
from cyroot.native.levenberg_marquardt cimport LevenbergMarquardt

cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil

cdef struct Parameters:
    double* matrix
    double* target
    size_t n
    size_t m
    size_t calls[2]
    size_t gil_held
    int fail_stage

cdef int residual(const double* x, size_t n, size_t m, double* out,
                  void* context) noexcept nogil:
    cdef Parameters* p = <Parameters*>context
    cdef size_t i,j
    p.calls[0] += 1
    if PyGILState_Check():p.gil_held += 1
    if p.fail_stage == 1:return -41
    if n != p.n or m != p.m:return -99
    for i in range(m):
        out[i] = -p.target[i]
        for j in range(n):out[i] += p.matrix[i+m*j]*x[j]
    return 0

cdef int jacobian(const double* x, size_t n, size_t m, double* out,
                  void* context) noexcept nogil:
    cdef Parameters* p = <Parameters*>context
    cdef size_t i
    p.calls[1] += 1
    if PyGILState_Check():p.gil_held += 1
    if p.fail_stage == 2:return -42
    if n != p.n or m != p.m:return -99
    for i in range(n*m):out[i] = p.matrix[i]
    return 0

cdef class Owner:
    cdef Parameters params
    cdef vector[double] matrix,target
    cdef object __weakref__
    def __init__(self,matrix,target):
        cdef size_t i,j
        self.params.m = len(matrix)
        self.params.n = len(matrix[0])
        self.matrix.resize(self.params.n*self.params.m)
        self.target.resize(self.params.m)
        for i in range(self.params.m):
            self.target[i] = target[i]
            for j in range(self.params.n):
                self.matrix[i+self.params.m*j] = matrix[i][j]
        self.params.matrix,self.params.target = self.matrix.data(),self.target.data()
    def callbacks(self):
        return (make_rectangular_callback(residual,self.params.n,self.params.m,
                                         <void*>&self.params,self),
                make_rectangular_callback(jacobian,self.params.n,self.params.m,
                                          <void*>&self.params,self,True))
    @property
    def counters(self):return (self.params.calls[0],self.params.calls[1],self.params.gil_held)
    @property
    def fail_stage(self):return self.params.fail_stage
    @fail_stage.setter
    def fail_stage(self,value):self.params.fail_stage = value


def run_raw(LevenbergMarquardt state):
    """Exercise the installed declarations and numerical loop without the GIL."""
    state._enter()
    try:
        with nogil:state._run()
        state._raise_callback_error()
        return state._snapshot(state._state,state._x,state._f)
    finally:state._release()


def reserved_probes(LevenbergMarquardt state, objective, callback, point):
    """All three entry points share a reservation during native execution."""
    state._enter()
    try:
        results=[]
        for function in (lambda:state.run(),lambda:objective.residual(point),
                         lambda:callback.evaluate(point)):
            try:function()
            except RuntimeError:results.append(True)
            else:results.append(False)
        return tuple(results)
    finally:state._release()
