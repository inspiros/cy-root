"""Small external native tensor consumer: raw descriptor and plan paths."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from cyroot.native.rectangular cimport make_rectangular_callback
from cyroot.native.higher_mixed cimport TensorDifferencePlan
from cyroot.native.higher_mixed_ops cimport (TensorDifferenceConfig,
                                            compute_tensor_difference)

cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil

cdef struct Parameters:
    size_t calls
    size_t gil_held
    Py_ssize_t fail_after
    Py_ssize_t partial_after
    int failure_code

cdef int polynomial(const double* x, size_t n, size_t m, double* out,
                    void* context) noexcept nogil:
    cdef Parameters* p = <Parameters*>context
    cdef size_t o, limit = m
    cdef double value
    p.calls += 1
    if PyGILState_Check():
        p.gil_held += 1
    if p.fail_after >= 0 and p.calls > <size_t>p.fail_after:
        return p.failure_code
    if n != 2:
        return -101
    if p.partial_after >= 0 and p.calls > <size_t>p.partial_after:
        limit = m-1
    value = x[0]*x[0]*x[0]+2*x[0]*x[0]*x[1]+3*x[0]*x[1]*x[1]+5*x[1]*x[1]*x[1]
    for o in range(limit):
        out[o] = 11.+o+(o+1)*value
    return 0

cdef class ContextOwner:
    cdef Parameters parameters
    cdef size_t m

    def __init__(self, m=3, fail_after=-1, partial_after=-1, failure_code=-43):
        self.m = m
        self.parameters.calls = self.parameters.gil_held = 0
        self.parameters.fail_after = fail_after
        self.parameters.partial_after = partial_after
        self.parameters.failure_code = failure_code

    def callback(self):
        return make_rectangular_callback(polynomial, 2, self.m,
                                         <void*>&self.parameters, self)

    @property
    def calls(self):
        return self.parameters.calls

    @property
    def gil_held(self):
        return self.parameters.gil_held


def allocation(TensorDifferencePlan plan):
    cdef TensorDifferenceConfig c
    plan.copy_config(&c)
    return (c.n,c.m,c.order,c.components,c.samples,c.terms,c.tensor_size,
            c.input_size,c.nodes,c.work,<size_t>c.steps,<size_t>c.indices,
            <size_t>c.starts,<size_t>c.term_samples,<size_t>c.offsets,<size_t>c.scatter)


def compute_direct(TensorDifferencePlan plan, point=(.5,1.), base=None,
                   fail_after=-1, partial_after=-1, initial_calls=0,
                   capacity=None, output_capacity=None, use_config=False,
                   config_overrides=None, repeats=1):
    cdef TensorDifferenceConfig c
    plan.copy_config(&c)
    cdef size_t n=c.n, m=c.m, needed=c.work, tensor=c.tensor_size
    cdef size_t count=needed if capacity is None else capacity
    cdef size_t output_count=tensor if output_capacity is None else output_capacity
    cdef size_t trial=c.samples*m
    cdef vector[double] x,f0,work,output
    cdef size_t calls=initial_calls, i, run, output_address, work_address
    cdef int status=-1, code=123
    cdef double value
    cdef ContextOwner owner=ContextOwner(m,fail_after,partial_after,-19)
    if n != 2 or len(point) != n or (base is not None and len(base) != m):
        raise ValueError('Fixture requires a two-input plan and matching spans.')
    x.resize(n)
    f0.resize(m)
    work.resize(needed+2)
    output.resize(tensor+2)
    for i in range(n):
        x[i]=point[i]
    value=x[0]*x[0]*x[0]+2*x[0]*x[0]*x[1]+3*x[0]*x[1]*x[1]+5*x[1]*x[1]*x[1]
    for i in range(m):
        f0[i]=11.+i+(i+1)*value if base is None else base[i]
    for i in range(needed+2):
        work[i]=9876543.
    for i in range(tensor+2):
        output[i]=1234567.
    output_address,work_address=<size_t>&output[1],<size_t>&work[1]
    if config_overrides is not None:
        for key,value_override in config_overrides.items():
            if key == 'n': c.n=value_override
            elif key == 'm': c.m=value_override
            elif key == 'order': c.order=value_override
            elif key == 'components': c.components=value_override
            elif key == 'samples': c.samples=value_override
            elif key == 'terms': c.terms=value_override
            elif key == 'tensor_size': c.tensor_size=value_override
            elif key == 'input_size': c.input_size=value_override
            elif key == 'nodes': c.nodes=value_override
            elif key == 'work': c.work=value_override
            elif key == 'scheme': c.scheme=value_override
            else: raise ValueError('Unknown tensor configuration field.')
    for run in range(repeats):
        if use_config:
            with nogil:
                status=compute_tensor_difference(&c,&x[0],&f0[0],&output[1],
                    output_count,&work[1],count,polynomial,
                    <void*>&owner.parameters,&calls,&code)
        else:
            with nogil:
                status=plan.compute(&x[0],&f0[0],&output[1],output_count,
                    &work[1],count,polynomial,<void*>&owner.parameters,&calls,&code)
    return dict(status=status,calls=calls,callback_code=code,
                flat=tuple(output[i+1] for i in range(tensor)),
                restored_point=tuple(work[1+trial+i] for i in range(n)),
                guards=(output[0],output[tensor+1],work[0],work[needed+1]),
                untouched_output=all(output[i] == 1234567. for i in range(tensor+2)),
                untouched_workspace=all(work[i] == 9876543. for i in range(needed+2)),
                stable=(output_address == <size_t>&output[1] and
                        work_address == <size_t>&work[1] and work.size() == needed+2),
                gil_held=owner.parameters.gil_held)
