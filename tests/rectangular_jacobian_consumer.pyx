"""External rectangular finite-difference consumer with bounded native spans."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from cyroot.native.derivatives cimport RectangularJacobianPlan
from cyroot.native.rectangular cimport make_rectangular_callback

cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil

cdef struct State:
    size_t calls, gil_held
    Py_ssize_t fail_after, partial_after
    int code
    double scale

cdef int affine(const double* x, size_t n, size_t m, double* out,
                void* context) noexcept nogil:
    cdef State* state = <State*>context
    cdef size_t i, j
    state.calls += 1
    if PyGILState_Check():
        state.gil_held += 1
    if state.fail_after >= 0 and state.calls > <size_t>state.fail_after:
        return state.code
    for i in range(m):
        if state.partial_after >= 0 and state.calls > <size_t>state.partial_after and i == m-1:
            continue
        out[i] = 0.
        for j in range(n):
            out[i] += state.scale*(10*(i+1)+j+1)*x[j]
    return 0

cdef class Owner:
    cdef State state
    cdef object __weakref__

    def __init__(self, fail_after=-1, code=-23, partial_after=-1, scale=1.):
        self.state.calls = self.state.gil_held = 0
        self.state.fail_after = fail_after
        self.state.partial_after = partial_after
        self.state.code = code
        self.state.scale = scale

    def callback(self, n, m):
        return make_rectangular_callback(affine, n, m, <void*>&self.state, self)

    @property
    def counters(self):
        return self.state.calls, self.state.gil_held

def compute(RectangularJacobianPlan plan, point, base=None, fail_after=-1,
            code=-23, partial_after=-1, scale=1.):
    cdef size_t n = plan.input_dimension, m = plan.output_dimension
    cdef size_t count = n*m, work_count = 3*n+2*m, i, j
    cdef vector[double] x, f0, out, work
    cdef size_t calls = 0
    cdef int callback_code = 0, status
    cdef Owner owner = Owner(fail_after, code, partial_after, scale)
    x.resize(n); f0.resize(m); out.resize(count+2); work.resize(work_count+2)
    for j in range(n): x[j] = point[j]
    for i in range(m):
        f0[i] = 0.
        for j in range(n): f0[i] += scale*(10*(i+1)+j+1)*x[j]
        if base is not None: f0[i] = base[i]
    for i in range(count+2): out[i] = 1234567.
    for i in range(work_count+2): work[i] = 7654321.
    with nogil:
        status = plan.compute(x.data(), f0.data(), &out[1], &work[1],
                              affine, <void*>&owner.state, &calls, &callback_code)
    return dict(status=status, calls=calls, callback_code=callback_code,
                jacobian=tuple(tuple(out[1+i+j*m] for j in range(n)) for i in range(m)),
                point=tuple(x[i] for i in range(n)), base=tuple(f0[i] for i in range(m)),
                restored=tuple(work[1+i] for i in range(n)),
                guards=(out[0], out[count+1], work[0], work[work_count+1]),
                counters=owner.counters)
