# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Bounded native tensor-product finite differences on rounded sample points."""
from itertools import combinations_with_replacement, groupby, product
from math import comb
from typing import NamedTuple
from libc.limits cimport INT_MAX
from libc.math cimport isfinite, NAN
from .buffers cimport OwnedDoubleBuffer
from .hessians cimport _pack_vector, _parse_scheme, _copy_steps
from .higher_mixed_ops cimport compute_tensor_difference
from .rectangular cimport RectangularCallback, RectangularFunction
from .vector cimport VectorCallback
from .derivatives cimport ResidualFunction
from .tensor_callbacks cimport HessianCallback, RectangularHessianCallback
from .higher_derivatives import _checked_integer
from .scalar import NativeCallbackError


class TensorDerivativeEstimate(NamedTuple):
    """Copied dense derivative and total residual-attempt diagnostics."""
    derivative: tuple
    f_calls: int
    status: str
    callback_code: int


cdef struct _SquareBridge:
    ResidualFunction function
    void* context

cdef int _square_residual(const double* x, size_t n, size_t m, double* out,
                         void* context) noexcept nogil:
    cdef _SquareBridge* bridge = <_SquareBridge*>context
    return bridge.function(x, n, out, bridge.context)

cdef int _evaluate_native(const TensorDifferenceConfig* config,
    const double* x, double* base, double* out, double* work, bint cached,
    RectangularFunction function, void* context, size_t* calls, int* code,
    bint* base_failure) noexcept nogil:
    cdef size_t i
    base_failure[0] = False
    code[0] = 0
    for i in range(config.tensor_size):out[i] = NAN
    for i in range(config.n):
        if not isfinite(x[i]):return 1
    if not cached:
        for i in range(config.m):base[i] = NAN
        calls[0] += 1
        code[0] = function(x, config.n, config.m, base, context)
        if code[0] != 0:
            base_failure[0] = True
            return 3
    for i in range(config.m):
        if not isfinite(base[i]):return 1
    return compute_tensor_difference(config, x, base, out, config.tensor_size,
        work, config.work, function, context, calls, code)

cdef object _nested(const double* out, size_t n, size_t m, size_t remaining,
                     size_t flat, size_t stride, size_t output):
    cdef object value
    cdef size_t j
    if remaining == 0:return out[output+m*flat]
    if n == 1:
        value = out[output]
        for j in range(remaining):value = (value,)
        return value
    return tuple([_nested(out,n,m,remaining-1,flat+i*stride,stride*n,output)
                  for i in range(n)])


def _width(order, scheme):
    return order+1+(order%2 if scheme == 2 else 0) if order else 1


def _offset(slot, scheme):
    if scheme == 0:return slot
    if scheme == 1:return -slot
    return (slot+1)//2 if slot%2 else -(slot//2)


cdef class TensorDifferencePlan:
    """Immutable dense derivatives of order >=3 for F:R^n -> R^m.

    Native byte limits cover retained graph/steps plus one native evaluation;
    Python construction/export objects and allocator overhead are additional.
    Stencil visits are separately bounded before enumeration. No NumPy is used.
    """
    def __init__(self, input_dimension, output_dimension, order,
                 scheme='central', step=None, *, max_nodes=65,
                 max_terms=1_000_000, memory_bytes=64*1024*1024):
        cdef size_t i, j
        cdef OwnedDoubleBuffer* steps = NULL
        if self._initialized or self._constructing:
            raise RuntimeError('Tensor difference plans cannot be reinitialized.')
        self._constructing = True
        try:
            n = _checked_integer(input_dimension,'input_dimension')
            m = _checked_integer(output_dimension,'output_dimension')
            degree = _checked_integer(order,'order')
            node_limit = _checked_integer(max_nodes,'max_nodes')
            term_limit = _checked_integer(max_terms,'max_terms')
            budget = _checked_integer(memory_bytes,'memory_bytes')
            scheme_code = _parse_scheme(scheme)
            if n == 0 or m == 0 or n > INT_MAX or m > INT_MAX:
                raise ValueError('Input/output dimensions must be positive and fit C int.')
            if degree < 3 or degree > INT_MAX:
                raise ValueError('TensorDifferencePlan requires order >=3 fitting C int; use Jacobian/Hessian plans for lower orders.')
            nodes = _width(degree,scheme_code)
            if nodes > node_limit:raise ValueError('Requested stencil exceeds max_nodes.')
            # Bound degree before loops or exponentiation, using integer arithmetic.
            coefficient_work = degree*nodes+degree+nodes*(degree+1)+2*nodes
            maximum = int(OwnedDoubleBuffer.max_size())
            if coefficient_work > min(maximum,budget//sizeof(double)):
                raise ValueError('Requested tensor scratch exceeds memory_bytes or native limits.')
            input_size = 1
            for unused in range(degree):
                input_size *= n
                if input_size > min(maximum,budget//sizeof(double))//m:
                    raise ValueError('Dense derivative output exceeds memory_bytes or native limits.')
            tensor_size = m*input_size
            components = comb(n+degree-1,degree)
            # Coefficient of (sum_r width(r)*z**r)**n; saturated at term limit.
            counts = [1]+[0]*degree
            for unused in range(n):
                updated = [0]*(degree+1)
                for total in range(degree+1):
                    value = 0
                    for r in range(total+1):
                        value += counts[total-r]*_width(r,scheme_code)
                        if value > term_limit:
                            value = term_limit+1
                            break
                    updated[total] = value
                counts = updated
            terms = counts[degree]
            if terms > term_limit:raise ValueError('Derivative planning exceeds max_terms.')
            step_count = n if step is not None else 0
            fixed_metadata = (components*degree+components+1+terms+input_size)*sizeof(size_t)
            fixed_work = 2*n+components*m+coefficient_work
            fixed_bytes = fixed_metadata+(n+m+tensor_size+fixed_work+step_count)*sizeof(double)
            sample_bytes = n*sizeof(int)+m*sizeof(double)
            if fixed_bytes > budget or fixed_work > maximum:
                raise ValueError('Requested tensor exceeds memory_bytes or native limits.')
            sample_limit = (budget-fixed_bytes)//sample_bytes
            if sample_limit == 0:raise ValueError('Requested sample cache exceeds memory_bytes.')
            if (components*degree > self._indices.max_size() or
                    components+1 > self._starts.max_size() or terms > self._term_samples.max_size() or
                    input_size > self._scatter.max_size()):
                raise ValueError('Requested graph exceeds native limits.')
            self._indices.resize(components*degree)
            self._starts.resize(components+1)
            self._term_samples.resize(terms)
            self._scatter.resize(input_size)
            sample_ids = {():0}
            component_ids = {}
            term_index = 0
            for component,index_tuple in enumerate(combinations_with_replacement(range(n),degree)):
                component_ids[index_tuple] = component
                self._starts[component] = term_index
                for j in range(degree):self._indices[component*degree+j] = index_tuple[j]
                groups = [(axis,len(tuple(items))) for axis,items in groupby(index_tuple)]
                widths = [_width(r,scheme_code) for axis,r in groups]
                for reversed_choices in product(*(range(w) for w in reversed(widths))):
                    choices = tuple(reversed(reversed_choices))
                    key = tuple((axis,_offset(slot,scheme_code))
                                for (axis,r),slot in zip(groups,choices) if slot != 0)
                    sample = sample_ids.get(key)
                    if sample is None:
                        sample = len(sample_ids)
                        if sample >= sample_limit:
                            raise ValueError('Requested sample cache exceeds memory_bytes.')
                        sample_ids[key] = sample
                    self._term_samples[term_index] = sample
                    term_index += 1
            if term_index != terms:raise RuntimeError('Internal stencil count mismatch.')
            self._starts[components] = terms
            samples = len(sample_ids)
            if samples*n > self._offsets.max_size():raise ValueError('Sample offsets exceed native limits.')
            self._offsets.resize(samples*n)
            for i in range(samples*n):self._offsets[i] = 0
            for key,sample in sample_ids.items():
                for axis,offset in key:self._offsets[sample*n+axis] = offset
            for flat in range(input_size):
                digits, remaining = [],flat
                for unused in range(degree):
                    digits.append(remaining%n)
                    remaining //= n
                self._scatter[flat] = component_ids[tuple(sorted(digits))]
            work = samples*m+fixed_work
            evaluation = n+m+tensor_size+work
            metadata = fixed_metadata+samples*n*sizeof(int)
            if evaluation > maximum or (evaluation+step_count)*sizeof(double)+metadata > budget:
                raise ValueError('Requested tensor exceeds memory_bytes or native limits.')
            steps = _copy_steps(n,step)
            self._n,self._m,self._order = n,m,degree
            self._components,self._samples,self._terms = components,samples,terms
            self._input_size,self._tensor_size,self._nodes = input_size,tensor_size,nodes
            self._work,self._evaluation_size,self._step_count = work,evaluation,step_count
            self._metadata_bytes = metadata
            self._memory_bytes = (evaluation+step_count)*sizeof(double)+metadata
            self._scheme,self._steps = scheme_code,steps
            steps = NULL
            self._initialized = True
        finally:
            if steps != NULL:del steps
            self._constructing = False

    def __dealloc__(self):
        if self._steps != NULL:del self._steps

    cdef void copy_config(self, TensorDifferenceConfig* config) noexcept nogil:
        config.n,config.m,config.order = self._n,self._m,self._order
        config.components,config.samples,config.terms = self._components,self._samples,self._terms
        config.input_size,config.tensor_size = self._input_size,self._tensor_size
        config.nodes,config.work,config.scheme = self._nodes,self._work,self._scheme
        config.steps = NULL
        if self._steps != NULL:config.steps = self._steps.data()
        config.indices,config.starts = self._indices.data(),self._starts.data()
        config.term_samples,config.offsets = self._term_samples.data(),self._offsets.data()
        config.scatter = self._scatter.data()

    cdef int compute(self, const double* x, const double* f0,
        double* out, size_t output_capacity, double* workspace, size_t capacity,
        RectangularFunction function, void* context, size_t* calls,
        int* callback_code) noexcept nogil:
        cdef TensorDifferenceConfig config
        callback_code[0] = 0
        if not self._initialized:return 2
        self.copy_config(&config)
        return compute_tensor_difference(&config,x,f0,out,output_capacity,workspace,
            capacity,function,context,calls,callback_code)

    def evaluate(self,function,x,f_x=None):
        cdef RectangularCallback rectangular = None
        cdef VectorCallback square = None
        cdef OwnedDoubleBuffer* storage = NULL
        cdef TensorDifferenceConfig config
        cdef RectangularFunction pointer
        cdef void* context
        cdef _SquareBridge bridge
        cdef double* point
        cdef double* base
        cdef double* out
        cdef double* work
        cdef size_t calls = 0
        cdef int code = 0,status
        cdef bint base_failure = False,cached = f_x is not None
        if self._constructing or not self._initialized:
            raise RuntimeError('Tensor difference plan is not initialized.')
        if isinstance(function,(HessianCallback,RectangularHessianCallback)):
            raise TypeError('function requires a residual callback, not a Hessian provider.')
        if isinstance(function,VectorCallback):
            square = function
            if square._function == NULL or square._constructing:
                raise RuntimeError('Residual callback adapter is not initialized.')
            if self._n != self._m or square._dimension != self._n or square._jacobian:
                raise ValueError('Square residual callback dimensions must match plan.')
            if square._active:raise RuntimeError('This callback adapter is already active.')
            square._active,square._error = True,None
            bridge.function,bridge.context = <ResidualFunction>square._function,square._context
            pointer,context = _square_residual,&bridge
        else:
            rectangular = function if isinstance(function,RectangularCallback) else RectangularCallback(function,self._n,self._m)
            if rectangular._function == NULL or rectangular._constructing:
                raise RuntimeError('Residual callback adapter is not initialized.')
            if rectangular._n != self._n or rectangular._m != self._m or rectangular._jacobian:
                raise ValueError('Residual callback dimensions must match plan.')
            rectangular._enter()
            pointer,context = rectangular._function,rectangular._context
        try:
            self.copy_config(&config)
            storage = new OwnedDoubleBuffer(self._evaluation_size)
            point = storage.data()
            base,out = point+self._n,point+self._n+self._m
            work = out+self._tensor_size
            _pack_vector(x,point,self._n)
            if cached:_pack_vector(f_x,base,self._m)
            with nogil:
                status = _evaluate_native(&config,point,base,out,work,cached,pointer,
                    context,&calls,&code,&base_failure)
            estimate = TensorDerivativeEstimate(tuple([_nested(out,self._n,self._m,self._order,0,1,i)
                for i in range(self._m)]),calls,('ok','nonfinite','breakdown','callback_failure')[status],code)
            if square is not None and square._error is not None:
                error = square._error
                square._error = None
                raise error
            if rectangular is not None and rectangular._error is not None:
                error = rectangular._error
                rectangular._error = None
                raise error
            if status == 3:raise NativeCallbackError(code,estimate,'function' if base_failure else 'derivative')
            return estimate
        finally:
            if storage != NULL:del storage
            if square is not None:square._active = False
            if rectangular is not None:rectangular._release()

    @property
    def input_dimension(self):return self._n
    @property
    def output_dimension(self):return self._m
    @property
    def order(self):return self._order
    @property
    def scheme(self):return ('forward','backward','central')[self._scheme]
    @property
    def step(self):return None if self._steps == NULL else tuple([self._steps.get(i) for i in range(self._n)])
    @property
    def shape(self):return (int(self._m),)+(int(self._n),)*int(self._order)
    @property
    def sample_count(self):return int(self._samples)-1 if self._samples else 0
    @property
    def storage_bytes(self):return self._metadata_bytes+self._step_count*sizeof(double)
    @property
    def workspace_size(self):return self._work
    @property
    def workspace_bytes(self):return self._work*sizeof(double)
    @property
    def output_bytes(self):return self._tensor_size*sizeof(double)
    @property
    def evaluation_bytes(self):return self._evaluation_size*sizeof(double)
    @property
    def memory_bytes(self):return self._memory_bytes
