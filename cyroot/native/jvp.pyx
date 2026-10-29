# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Matrix-free Jacobian-vector products for Python and compiled consumers."""
from typing import NamedTuple
from libc.limits cimport INT_MAX
from libc.math cimport isfinite, NAN
from cpython.buffer cimport PyObject_CheckBuffer
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from .buffers cimport OwnedDoubleBuffer, VectorView
from ._buffers cimport DoubleBuffer
from .vector cimport VectorCallback
from .rectangular cimport RectangularCallback
from .scalar import _nonnegative_integer, NativeCallbackError
from .derivatives cimport DIFF_OK, DIFF_NONFINITE, DIFF_BREAKDOWN, DIFF_CALLBACK_FAILURE

__all__=['JacobianVectorCallback','JacobianVectorPlan','JacobianVectorEstimate']

cdef extern from "jvp.hpp" namespace "cyroot::native" nogil:
    int jvp_compute(size_t,size_t,int,double,const double*,const double*,const double*,
                    double*,double*,RectangularFunction,void*,size_t*,int*) noexcept

class JacobianVectorEstimate(NamedTuple):
    product: tuple
    f_calls: int
    status: str
    callback_code: int

cdef void _fill(double* out,size_t count,double value) noexcept nogil:
    cdef size_t i
    for i in range(count):out[i]=value

cdef void _pack(object source,double* out,size_t count) except *:
    cdef DoubleBuffer guard
    cdef VectorView view
    cdef size_t i
    if isinstance(source,(str,bytes,bytearray)):raise TypeError('Expected a numeric vector.')
    if PyObject_CheckBuffer(source):
        guard=DoubleBuffer(source);view=guard.vector_view()
        if view.size()!=count:raise ValueError('Vector length must match the declared dimension.')
        for i in range(count):out[i]=view.get(i)
    else:
        if len(source)!=<Py_ssize_t>count:raise ValueError('Vector length must match the declared dimension.')
        for i in range(count):out[i]=source[i]

cdef bint _shape(size_t n,size_t m) noexcept nogil:
    cdef size_t maximum=OwnedDoubleBuffer.max_size()
    return n>0 and m>0 and n<=INT_MAX and m<=INT_MAX and n<=maximum//2 and m<=maximum-2*n

cdef int _python_product(const double* x,const double* v,size_t n,size_t m,
                          double* out,void* context) noexcept nogil:
    cdef size_t i
    with gil:
        callback=<JacobianVectorCallback><object>context
        try:
            value=callback._python_function(tuple([x[i] for i in range(n)]),tuple([v[i] for i in range(n)]))
            _pack(value,out,m)
        except BaseException as error:
            callback._error=error
            return 1
    return 0

cdef class JacobianVectorCallback:
    """A supplied product J(x)v with n inputs and m outputs, without storing J."""
    def __init__(self,function,input_dimension,output_dimension):
        if self._function!=NULL or self._constructing:raise RuntimeError('Callback adapters cannot be reinitialized.')
        self._constructing=True
        try:
            n=_nonnegative_integer(input_dimension,'input_dimension')
            m=_nonnegative_integer(output_dimension,'output_dimension')
            if not _shape(n,m):raise ValueError('Positive dimensions must fit native capacities and C int.')
            if not callable(function):raise TypeError('function must be callable.')
            self._n,self._m=n,m
            self._python_function=function;self._context=<void*>self;self._function=_python_product
        finally:self._constructing=False

    @classmethod
    def from_capsule(cls,capsule,keepalive=None):
        cdef cyroot_jvp_descriptor* descriptor
        if not PyCapsule_IsValid(capsule,b'cyroot.jacobian_vector_callback.v1'):
            raise ValueError('Expected a cyroot.jacobian_vector_callback.v1 capsule.')
        descriptor=<cyroot_jvp_descriptor*>PyCapsule_GetPointer(capsule,b'cyroot.jacobian_vector_callback.v1')
        if descriptor.struct_size<sizeof(cyroot_jvp_descriptor) or descriptor.abi_version!=1:
            raise ValueError('Unsupported Jacobian-vector callback size or ABI version.')
        return make_jacobian_vector_callback(descriptor.function,descriptor.input_dimension,
                                            descriptor.output_dimension,descriptor.context,(capsule,keepalive))

    cdef void _enter(self) except *:
        if self._constructing or self._function==NULL:raise RuntimeError('Callback is not initialized.')
        if self._active:raise RuntimeError('This callback adapter is already active.')
        self._active=True;self._error=None

    cdef void _leave(self) noexcept:self._active=False

    cdef int evaluate_raw(self,const double* x,const double* direction,double* out) noexcept nogil:
        cdef int code
        _fill(out,self._m,NAN)
        code=self._function(x,direction,self._n,self._m,out,self._context)
        if code!=0:_fill(out,self._m,NAN)
        return code

    def evaluate(self,point,direction,*,workspace_bytes=64*1024*1024):
        """Copy inputs, evaluate J(x)v, and return a copied output tuple.

        Inputs must be finite. Native callbacks run without the GIL; Python
        callbacks receive copied tuples. Nonfinite callback output is returned
        unchanged so consumers can apply their own numerical failure policy.
        """
        cdef OwnedDoubleBuffer* storage=NULL
        cdef double* data
        cdef double* out
        cdef size_t i
        cdef int code
        self._enter()
        try:
            limit=_nonnegative_integer(workspace_bytes,'workspace_bytes')
            count=2*int(self._n)+int(self._m)
            if count*sizeof(double)>limit:raise ValueError('Callback storage exceeds workspace_bytes.')
            storage=new OwnedDoubleBuffer(count);data=storage.data();out=data+2*self._n
            _pack(point,data,self._n);_pack(direction,data+self._n,self._n)
            for i in range(2*self._n):
                if not isfinite(data[i]):raise ValueError('Point and direction must be finite.')
            with nogil:code=self.evaluate_raw(data,data+self._n,out)
            if self._error is not None:
                error=self._error;self._error=None;raise error
            result=tuple([out[i] for i in range(self._m)])
            if code!=0:raise NativeCallbackError(code,result,'jacobian_vector_product')
            return result
        finally:
            if storage!=NULL:del storage
            self._leave()

    @property
    def input_dimension(self):return self._n
    @property
    def output_dimension(self):return self._m
    @property
    def native(self):return bool(self._native)

cdef JacobianVectorCallback make_jacobian_vector_callback(JacobianVectorFunction function,
        size_t n,size_t m,void* context,object keepalive=None):
    if function==NULL:raise ValueError('Native product function must not be NULL.')
    if not _shape(n,m):raise ValueError('Positive dimensions must fit native capacities and C int.')
    cdef JacobianVectorCallback callback=JacobianVectorCallback.__new__(JacobianVectorCallback)
    callback._n,callback._m=n,m
    callback._function=function;callback._context=context;callback._keepalive=keepalive;callback._native=True
    return callback

cdef int _rectangular_sample(const double* x,size_t n,size_t m,double* out,void* context) noexcept nogil:
    return (<RectangularCallback><object>context).evaluate_raw(x,out)

cdef int _square_sample(const double* x,size_t n,size_t m,double* out,void* context) noexcept nogil:
    return (<VectorCallback><object>context)._function(x,n,out,(<VectorCallback><object>context)._context)

cdef class JacobianVectorPlan:
    """Immutable directional finite differences using O(n+m) scratch storage.

    step=None uses a relative spatial step along v/||v||_infinity; an explicit
    positive step is an absolute displacement along that normalized direction.
    Forward/backward need one shifted residual; central needs two. All use F(x),
    supplied as f_x or evaluated once. A zero direction needs no residual calls.
    Collapsed active coordinates report breakdown before sampling. Poorly scaled
    problems may need rescaling or an explicit step; no error bound is claimed.
    """
    def __init__(self,input_dimension,output_dimension,scheme='central',step=None,*,memory_bytes=64*1024*1024):
        cdef double value
        if self._initialized or self._constructing:raise RuntimeError('Plans cannot be reinitialized.')
        self._constructing=True
        try:
            n=_nonnegative_integer(input_dimension,'input_dimension');m=_nonnegative_integer(output_dimension,'output_dimension')
            limit=_nonnegative_integer(memory_bytes,'memory_bytes')
            if not _shape(n,m):raise ValueError('Positive dimensions must fit native capacities and C int.')
            if scheme not in ('forward','backward','central'):raise ValueError('Unknown difference scheme.')
            code=0 if scheme=='forward' else 1 if scheme=='backward' else 2
            value=0. if step is None else float(step)
            if not isfinite(value) or (step is not None and value<=0.):raise ValueError('step must be finite and positive.')
            work=2*n+3*m;evaluation=2*n+2*m+work
            if evaluation>OwnedDoubleBuffer.max_size() or evaluation*sizeof(double)>limit:
                raise ValueError('Directional evaluation exceeds memory_bytes or native limits.')
            self._n,self._m=n,m;self._scheme=code;self._step=value
            self._work_size=work;self._evaluation_size=evaluation;self._memory_bytes=evaluation*sizeof(double)
            self._initialized=True
        finally:self._constructing=False

    cdef int compute(self,const double* x,const double* direction,const double* base,
                     double* out,double* work,RectangularFunction function,void* context,
                     size_t* calls,int* callback_code) noexcept nogil:
        cdef int status
        callback_code[0]=0
        if not self._initialized or function==NULL:
            _fill(out,self._m,NAN);return DIFF_BREAKDOWN
        status=jvp_compute(self._n,self._m,self._scheme,self._step,x,direction,base,
                           out,work,function,context,calls,callback_code)
        if status!=DIFF_OK:_fill(out,self._m,NAN)
        return status

    def evaluate(self,function,x,direction,f_x=None):
        """Return a copied product/status; callback failures propagate as usual."""
        cdef RectangularCallback rectangular=None
        cdef VectorCallback square=None
        cdef RectangularFunction sample
        cdef void* context
        cdef OwnedDoubleBuffer* storage=NULL
        cdef double* data
        cdef double* base
        cdef double* out
        cdef double* work
        cdef const double* cached=NULL
        cdef size_t i,calls=0
        cdef int code=0,status
        if not self._initialized:raise RuntimeError('Plan is not initialized.')
        if isinstance(function,VectorCallback):
            square=function
            if square._function==NULL or square._constructing:raise RuntimeError('Residual callback is not initialized.')
            if square._jacobian or square._dimension!=self._n or self._n!=self._m:
                raise ValueError('Residual callback dimensions/kind must match the plan.')
            if square._active:raise RuntimeError('This callback adapter is already active.')
            square._active=True;square._error=None
            sample=_square_sample;context=<void*>square
        else:
            rectangular=function if isinstance(function,RectangularCallback) else RectangularCallback(function,self._n,self._m)
            if rectangular._jacobian or rectangular._n!=self._n or rectangular._m!=self._m:
                raise ValueError('Residual callback dimensions/kind must match the plan.')
            rectangular._enter();sample=_rectangular_sample;context=<void*>rectangular
        try:
            storage=new OwnedDoubleBuffer(self._evaluation_size);data=storage.data()
            base=data+2*self._n;out=base+self._m;work=out+self._m
            _pack(x,data,self._n);_pack(direction,data+self._n,self._n)
            if f_x is not None:_pack(f_x,base,self._m);cached=base
            with nogil:status=self.compute(data,data+self._n,cached,out,work,sample,context,&calls,&code)
            result=JacobianVectorEstimate(tuple([out[i] for i in range(self._m)]),calls,
                'ok' if status==DIFF_OK else 'nonfinite' if status==DIFF_NONFINITE else
                'breakdown' if status==DIFF_BREAKDOWN else 'callback_failure',code)
            error=square._error if square is not None else rectangular._error
            if square is not None:square._error=None
            else:rectangular._error=None
            if error is not None:raise error
            if status==DIFF_CALLBACK_FAILURE:raise NativeCallbackError(code,result,'function')
            return result
        finally:
            if storage!=NULL:del storage
            if square is not None:square._active=False
            else:rectangular._leave()

    @property
    def input_dimension(self):return self._n
    @property
    def output_dimension(self):return self._m
    @property
    def scheme(self):return ('forward','backward','central')[self._scheme] if self._initialized else None
    @property
    def step(self):return None if self._step==0. else self._step
    @property
    def workspace_bytes(self):return self._work_size*sizeof(double)
    @property
    def memory_bytes(self):return self._memory_bytes
