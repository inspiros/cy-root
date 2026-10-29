# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Matrix-free Jacobian-transpose products for Python and compiled consumers."""
from libc.limits cimport INT_MAX
from libc.math cimport isfinite, NAN
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from .buffers cimport OwnedDoubleBuffer, VectorView
from ._buffers cimport DoubleBuffer, is_native_double_buffer
from .scalar import _nonnegative_integer, NativeCallbackError

__all__=['JacobianTransposeCallback']

cdef void _fill(double* out,size_t count,double value) noexcept nogil:
    cdef size_t i
    for i in range(count):out[i]=value

cdef void _pack(object source,double* out,size_t count) except *:
    cdef DoubleBuffer guard
    cdef VectorView view
    cdef size_t i
    if isinstance(source,(str,bytes,bytearray)):raise TypeError('Expected a numeric vector.')
    if is_native_double_buffer(source):
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
        callback=<JacobianTransposeCallback><object>context
        try:
            value=callback._python_function(tuple([x[i] for i in range(n)]),tuple([v[i] for i in range(m)]))
            _pack(value,out,n)
        except BaseException as error:
            callback._error=error
            return 1
    return 0

cdef class JacobianTransposeCallback:
    """A supplied product J(x)^T w for an m-by-n Jacobian, without storing J.

    input_dimension is the length n of x; output_dimension is the length m of
    the underlying residual F(x). The direction has length m and the returned
    product has length n. Native producers fill all n output entries.
    """
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
        cdef cyroot_vjp_descriptor* descriptor
        if not PyCapsule_IsValid(capsule,b'cyroot.jacobian_transpose_callback.v1'):
            raise ValueError('Expected a cyroot.jacobian_transpose_callback.v1 capsule.')
        descriptor=<cyroot_vjp_descriptor*>PyCapsule_GetPointer(capsule,b'cyroot.jacobian_transpose_callback.v1')
        if descriptor.struct_size<sizeof(cyroot_vjp_descriptor) or descriptor.abi_version!=1:
            raise ValueError('Unsupported Jacobian-transpose callback size or ABI version.')
        return make_jacobian_transpose_callback(descriptor.function,descriptor.input_dimension,
                                            descriptor.output_dimension,descriptor.context,(capsule,keepalive))

    cdef void _enter(self) except *:
        if self._constructing or self._function==NULL:raise RuntimeError('Callback is not initialized.')
        if self._active:raise RuntimeError('This callback adapter is already active.')
        self._active=True;self._error=None

    cdef void _release(self) noexcept:self._active=False

    cdef int evaluate_raw(self,const double* x,const double* direction,double* out) noexcept nogil:
        cdef int code
        _fill(out,self._n,NAN)
        code=self._function(x,direction,self._n,self._m,out,self._context)
        if code!=0:_fill(out,self._n,NAN)
        return code

    def evaluate(self,point,direction,*,workspace_bytes=64*1024*1024):
        """Copy inputs, evaluate J(x)^T w, and return a copied output tuple.

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
            storage=new OwnedDoubleBuffer(count);data=storage.data();out=data+self._n+self._m
            _pack(point,data,self._n);_pack(direction,data+self._n,self._m)
            for i in range(self._n+self._m):
                if not isfinite(data[i]):raise ValueError('Point and direction must be finite.')
            with nogil:code=self.evaluate_raw(data,data+self._n,out)
            if self._error is not None:
                error=self._error;self._error=None;raise error
            result=tuple([out[i] for i in range(self._n)])
            if code!=0:raise NativeCallbackError(code,result,'jacobian_transpose_product')
            return result
        finally:
            if storage!=NULL:del storage
            self._release()

    @property
    def input_dimension(self):return self._n
    @property
    def output_dimension(self):return self._m
    @property
    def native(self):return bool(self._native)

cdef JacobianTransposeCallback make_jacobian_transpose_callback(JacobianTransposeFunction function,
        size_t n,size_t m,void* context,object keepalive=None):
    if function==NULL:raise ValueError('Native product function must not be NULL.')
    if not _shape(n,m):raise ValueError('Positive dimensions must fit native capacities and C int.')
    cdef JacobianTransposeCallback callback=JacobianTransposeCallback.__new__(JacobianTransposeCallback)
    callback._n,callback._m=n,m
    callback._function=function;callback._context=context;callback._keepalive=keepalive;callback._native=True
    return callback
