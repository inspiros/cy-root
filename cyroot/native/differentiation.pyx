# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Explicit complex-step and adaptive Richardson derivative providers.

These are supplied-derivative callbacks, usable by existing solvers without
changing their numerical engines. Python functions acquire the GIL; native
callbacks, sampling, extrapolation and scratch management during a call do not.
"""
from typing import NamedTuple
from libc.math cimport isfinite, NAN, INFINITY, fabs, fmax
from libc.limits cimport INT_MAX
from libc.string cimport memcpy
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from .scalar cimport make_scalar_callback
from .scalar import NativeCallbackError
from .rectangular cimport make_rectangular_callback
from .vector cimport VectorCallback
from .complex_callbacks cimport make_complex_scalar_callback
from .higher_derivatives import _checked_integer
from .hessians cimport RectangularHessianConfig, compute_rectangular_hessian

__all__ = ['AdaptiveHessian', 'NumericalHessianEstimate', 'AdaptiveSecondDerivative', 'ComplexStepDerivative', 'ComplexStepJacobian', 'AdaptiveDerivative',
           'AdaptiveJacobian', 'ComplexVectorCallback', 'DerivativeEstimate',
           'NumericalJacobianEstimate', 'DifferentiationError']


class DerivativeEstimate(NamedTuple):
    derivative: float
    error: object
    f_calls: int
    status: str
    callback_code: int


class NumericalJacobianEstimate(NamedTuple):
    jacobian: tuple
    error: object
    f_calls: int
    status: str
    callback_code: int


class NumericalHessianEstimate(NamedTuple):
    hessian: tuple
    error: tuple
    f_calls: int
    status: str
    callback_code: int


class DifferentiationError(ArithmeticError):
    """An estimator did not produce an acceptable derivative for a solver."""
    def __init__(self, estimate):
        self.estimate = estimate
        super().__init__(f'Derivative estimation stopped with status {estimate.status}.')


def _status(code):
    return ('ok', 'nonfinite', 'breakdown', 'callback_failure', 'max_steps', 'roundoff')[code]


def _number(value, name, positive=False):
    if isinstance(value, (str, bytes, bytearray, bool)):
        raise TypeError(f'{name} must be a finite real number.')
    result = float(value)
    if not isfinite(result) or result < 0. or (positive and result == 0.):
        raise ValueError(f'{name} must be finite and '+('positive.' if positive else 'nonnegative.'))
    return result


def _settings(atol, rtol, max_steps):
    atol, rtol = _number(atol, 'atol'), _number(rtol, 'rtol')
    levels = _checked_integer(max_steps, 'max_steps')
    if not 2 <= levels <= 32:raise ValueError('max_steps must be between 2 and 32.')
    return atol, rtol, levels


cdef void _pack_point(object source,double* out,size_t count) except *:
    cdef size_t i
    if isinstance(source,(str,bytes,bytearray)) or len(source)!=<Py_ssize_t>count:
        raise ValueError('Point length must match input_dimension.')
    for i in range(count):out[i]=source[i]


cdef tuple _shape(object input_dimension, object output_dimension):
    n = _checked_integer(input_dimension, 'input_dimension')
    m = _checked_integer(output_dimension, 'output_dimension')
    if not 0 < n <= INT_MAX or not 0 < m <= INT_MAX:
        raise ValueError('Dimensions must be positive and fit C int.')
    return n, m


cdef ScalarCallback _private_real(object function):
    cdef ScalarCallback source
    if isinstance(function,_ScalarDifferentiation):raise TypeError('Supply a residual, not another derivative estimator.')
    if not isinstance(function, ScalarCallback):return ScalarCallback(function)
    source = function
    if source._function == NULL or source._active:
        raise RuntimeError('Residual callback is uninitialized or active.')
    if source._native:return make_scalar_callback(source._function, source._context, source)
    return ScalarCallback(source._python_function)


cdef ComplexScalarCallback _private_complex(object function):
    cdef ComplexScalarCallback source
    if not isinstance(function, ComplexScalarCallback):return ComplexScalarCallback(function)
    source = function
    if source._function == NULL or source._active:
        raise RuntimeError('Complex callback is uninitialized or active.')
    if source._native:return make_complex_scalar_callback(source._function, source._context, source)
    return ComplexScalarCallback(source._python_function)


cdef int _python_complex_vector(const double* real, const double* imag, size_t n,
        size_t m, double* out_real, double* out_imag, void* context) noexcept nogil:
    cdef size_t i
    cdef double complex item
    with gil:
        adapter = <ComplexVectorCallback><object>context
        try:
            value = adapter._python_function(tuple([complex(real[i], imag[i]) for i in range(n)]))
            if isinstance(value, (str, bytes, bytearray)) or len(value) != <Py_ssize_t>m:
                raise ValueError('Complex residual length must match output_dimension.')
            for i in range(m):
                item = value[i]
                out_real[i], out_imag[i] = item.real, item.imag
        except BaseException as error:
            adapter._error = error
            return 1
    return 0


cdef class ComplexVectorCallback:
    """Complex residual with n inputs/m outputs, using split arrays natively.

    Native producers cimport make_complex_vector_callback or supply a capsule
    named cyroot.complex_vector_callback.v1, declared in complex_callbacks.h.
    Contexts shared through different adapters need producer synchronization.
    """
    def __init__(self, function, input_dimension, output_dimension):
        if self._function != NULL or self._constructing:
            raise RuntimeError('Callback adapters cannot be reinitialized.')
        self._constructing = True
        try:
            n, m = _shape(input_dimension, output_dimension)
            if 2*(n+m) > OwnedDoubleBuffer.max_size():raise ValueError('Dimensions exceed native capacity.')
            if not callable(function):raise TypeError('function must be callable.')
            self._n, self._m = n, m
            self._python_function = function
            self._context = <void*>self
            self._function = _python_complex_vector
        finally:
            self._constructing = False

    @classmethod
    def from_capsule(cls, capsule, keepalive=None):
        cdef cyroot_complex_vector_descriptor* descriptor
        if not PyCapsule_IsValid(capsule, b'cyroot.complex_vector_callback.v1'):
            raise ValueError('Expected a cyroot.complex_vector_callback.v1 capsule.')
        descriptor = <cyroot_complex_vector_descriptor*>PyCapsule_GetPointer(capsule, b'cyroot.complex_vector_callback.v1')
        if descriptor.struct_size < sizeof(cyroot_complex_vector_descriptor) or descriptor.abi_version != 1:
            raise ValueError('Unsupported complex vector descriptor size or ABI version.')
        return make_complex_vector_callback(descriptor.function, descriptor.input_dimension,
            descriptor.output_dimension, descriptor.context, (capsule, keepalive))

    cdef void _enter(self) except *:
        if self._constructing or self._function == NULL:raise RuntimeError('Callback is not initialized.')
        if self._active:raise RuntimeError('This callback adapter is already active.')
        self._active, self._error = True, None

    cdef void _release(self) noexcept:
        self._active = False

    def evaluate(self, point, *, workspace_bytes=64*1024*1024):
        cdef OwnedDoubleBuffer* storage = NULL
        cdef double* xr
        cdef double* xi
        cdef double* fr
        cdef double* fi
        cdef size_t i
        cdef double complex z
        cdef int code
        self._enter()
        try:
            limit = _checked_integer(workspace_bytes, 'workspace_bytes')
            count = 2*(int(self._n)+int(self._m))
            if count*sizeof(double)>limit:raise ValueError('Complex callback storage exceeds workspace_bytes.')
            storage = new OwnedDoubleBuffer(count)
            xr=storage.data(); xi=xr+self._n; fr=xi+self._n; fi=fr+self._m
            if isinstance(point,(str,bytes,bytearray)) or len(point)!=<Py_ssize_t>self._n:
                raise ValueError('Point length must match input_dimension.')
            for i in range(self._n):
                z=point[i]; xr[i],xi[i]=z.real,z.imag
            for i in range(self._m):fr[i]=fi[i]=NAN
            with nogil:code=self._function(xr,xi,self._n,self._m,fr,fi,self._context)
            if self._error is not None:
                error=self._error; self._error=None; raise error
            if code:raise NativeCallbackError(code,None,'function')
            return tuple([complex(fr[i],fi[i]) for i in range(self._m)])
        finally:
            if storage != NULL:del storage
            self._release()

    @property
    def native(self):return bool(self._native)
    @property
    def input_dimension(self):return self._n
    @property
    def output_dimension(self):return self._m


cdef ComplexVectorCallback make_complex_vector_callback(ComplexVectorFunction function,
        size_t n, size_t m, void* context, object keepalive=None):
    if function == NULL:raise ValueError('Native callback must not be NULL.')
    _shape(n,m)
    if 2*(int(n)+int(m))>OwnedDoubleBuffer.max_size():raise ValueError('Dimensions exceed native capacity.')
    cdef ComplexVectorCallback adapter=ComplexVectorCallback.__new__(ComplexVectorCallback)
    adapter._n,adapter._m=n,m
    adapter._function,adapter._context=function,context
    adapter._keepalive,adapter._native=keepalive,True
    return adapter


cdef int _scalar_real(const double* x,size_t n,size_t m,double* out,void* context) noexcept nogil:
    return (<ScalarCallback><object>context)._function(x[0],out,(<ScalarCallback><object>context)._context)


cdef int _scalar_complex(const double* xr,const double* xi,size_t n,size_t m,
        double* fr,double* fi,void* context) noexcept nogil:
    return (<ComplexScalarCallback><object>context)._function(xr[0],xi[0],fr,fi,(<ComplexScalarCallback><object>context)._context)


cdef int _scalar_provider_error(void* context,int status) noexcept with gil:
    provider=<_ScalarDifferentiation><object>context
    error=provider._real._error if provider._adaptive else provider._complex._error
    if error is not None:
        provider._error=error
        if provider._adaptive:provider._real._error=None
        else:provider._complex._error=None
    elif status!=ESTIMATE_CALLBACK_FAILURE:
        provider._error=DifferentiationError(provider._estimate())
    return provider._report.callback_code if status==ESTIMATE_CALLBACK_FAILURE else -1000-status


cdef int _scalar_provider(double x,double* out,void* context) noexcept nogil:
    cdef int status=(<_ScalarDifferentiation><object>context)._compute(x)
    if status==ESTIMATE_OK:
        out[0]=(<_ScalarDifferentiation><object>context)._storage.data()[0]
        return 0
    out[0]=NAN
    return _scalar_provider_error(context,status)


cdef class _ScalarDifferentiation:
    def __init__(self):raise TypeError('Choose a concrete scalar derivative estimator.')

    cdef void _configure(self,object function,bint adaptive,object step,
                        object atol,object rtol,object max_steps,int order) except *:
        if (self._function != NULL or self._building or self._storage!=NULL or
                self._real is not None or self._complex is not None):
            raise RuntimeError('Derivative providers cannot be reinitialized.')
        self._building=True
        try:
            self._step=0. if step is None else _number(step,'step',True)
            self._atol,self._rtol,self._levels=_settings(atol,rtol,max_steps)
            self._adaptive=adaptive
            self._order=order
            if adaptive:self._real=_private_real(function)
            else:self._complex=_private_complex(function)
            self._storage=new OwnedDoubleBuffer(2+(self._levels+6 if adaptive else 4))
            self._report.status=-1
            self._context=<void*>self
            self._native=self._real._native if adaptive else self._complex._native
            self._function=_scalar_provider
        finally:
            self._building=False

    def __dealloc__(self):
        if self._storage != NULL:del self._storage

    cdef void _check_ready(self) except *:
        if self._building or self._function == NULL:raise RuntimeError('Derivative provider is not initialized.')
        if self._active:raise RuntimeError('This derivative provider is already active.')

    cdef int _compute(self,double x) noexcept nogil:
        cdef double* steps=NULL
        if self._step!=0.:steps=&self._step
        if self._adaptive:
            if self._order==2:
                return adaptive_second_derivative(x,steps,self._atol,self._rtol,self._levels,
                    self._storage.data(),self._storage.data()+1,self._storage.data()+2,
                    _scalar_real,<void*>self._real,self._report)
            return adaptive_jacobian(&x,1,1,steps,self._atol,self._rtol,self._levels,
                self._storage.data(),self._storage.data()+1,self._storage.data()+2,
                _scalar_real,<void*>self._real,self._report)
        return complex_step_jacobian(&x,1,1,steps,self._storage.data(),
            self._storage.data()+2,_scalar_complex,<void*>self._complex,self._report)

    cdef object _estimate(self):
        if self._report.status<0:raise RuntimeError('No derivative has been evaluated.')
        return DerivativeEstimate(self._storage.get(0),self._storage.get(1) if self._adaptive else None,
            self._report.calls,_status(self._report.status),self._report.callback_code)

    def evaluate(self,x):
        """Return a copied estimate; nonconvergence retains the best approximation.

        Python callback exceptions propagate. Native callback failures raise
        NativeCallbackError with this estimate. Other statuses are diagnostics.
        """
        cdef double point
        self._check_ready(); self._active=True
        try:
            point=x
            with nogil:self._compute(point)
            estimate=self._estimate()
            error=self._real._error if self._adaptive else self._complex._error
            if error is not None:
                if self._adaptive:self._real._error=None
                else:self._complex._error=None
                raise error
            if self._report.status==ESTIMATE_CALLBACK_FAILURE:
                raise NativeCallbackError(self._report.callback_code,estimate,'function')
            return estimate
        finally:
            self._active=False

    def __call__(self,x):
        estimate=self.evaluate(x)
        if estimate.status!='ok':raise DifferentiationError(estimate)
        return estimate.derivative

    @property
    def last_estimate(self):
        self._check_ready()
        return self._estimate()
    @property
    def step(self):return None if self._step==0. else self._step
    @property
    def max_steps(self):return int(self._levels) if self._adaptive else None
    @property
    def order(self):return self._order


cdef class ComplexStepDerivative:
    """First derivative of a real function via its holomorphic complex extension.

    function must accept complex arguments and preserve imaginary perturbations.
    abs/conjugation/real casts can silently give wrong answers; no analyticity
    test is attempted. step=None uses 1e-20*max(1,abs(x)); explicit step is a
    positive absolute displacement. No reliable error estimate is supplied.
    """
    def __init__(self,function,*,step=None):
        self._configure(function,False,step,0.,0.,2,1)


cdef class AdaptiveDerivative:
    """Bounded central differences with Richardson extrapolation for the first derivative.

    step=None starts at .1*max(1,abs(x)); a positive explicit step is absolute.
    Steps halve; max_steps is 2..32. Error estimates are adjacent-table
    differences, not rigorous bounds. Functions must be smooth/deterministic
    throughout the stencil. Solver use requires error <= atol+rtol*abs(value).
    """
    def __init__(self,function,*,step=None,atol=1e-10,rtol=1e-8,max_steps=10):
        self._configure(function,True,step,atol,rtol,max_steps,1)


cdef class AdaptiveSecondDerivative:
    """Scalar second derivative with bounded central Richardson refinement.

    Uses a quadratic interpolant on actual rounded offsets, then extrapolates
    across halved steps. Options, estimated errors, statuses and callback counts
    follow AdaptiveDerivative. A successful estimate is not a rigorous accuracy
    bound; cancellation/noise can be more severe for second derivatives.
    Pass this provider in a solver's second-derivative argument, e.g. Halley.
    """
    def __init__(self,function,*,step=None,atol=1e-10,rtol=1e-8,max_steps=10):
        self._configure(function,True,step,atol,rtol,max_steps,2)


cdef int _vector_real(const double* x,size_t n,size_t m,double* out,void* context) noexcept nogil:
    return (<RectangularCallback><object>context).evaluate_raw(x,out)


cdef int _square_real(const double* x,size_t n,size_t m,double* out,void* context) noexcept nogil:
    return (<VectorCallback><object>context)._function(x,n,out,(<VectorCallback><object>context)._context)


cdef RectangularCallback _private_vector(object function,size_t n,size_t m):
    cdef RectangularCallback real
    cdef VectorCallback square
    if isinstance(function,(_JacobianDifferentiation,RectangularHessianCallback)):
        raise TypeError('Supply a residual, not a derivative estimator.')
    if isinstance(function,RectangularCallback):
        real=function
        if real._function==NULL or real._active or real._constructing:
            raise RuntimeError('Residual callback is uninitialized or active.')
        if real._n!=n or real._m!=m or real._jacobian:
            raise ValueError('Residual dimensions/kind must match the estimator.')
        return make_rectangular_callback(real._function,n,m,real._context,real) if real._native else RectangularCallback(real._python_function,n,m)
    if isinstance(function,VectorCallback):
        square=function
        if square._function==NULL or square._active:
            raise RuntimeError('Residual callback is uninitialized or active.')
        if n!=m or square._dimension!=n or square._jacobian:
            raise ValueError('Square residual dimensions/kind must match the estimator.')
        return make_rectangular_callback(_square_real,n,m,<void*>square,square) if square._native else RectangularCallback(square._python_function,n,m)
    return RectangularCallback(function,n,m)


cdef int _vector_complex(const double* xr,const double* xi,size_t n,size_t m,
        double* fr,double* fi,void* context) noexcept nogil:
    return (<ComplexVectorCallback><object>context)._function(xr,xi,n,m,fr,fi,(<ComplexVectorCallback><object>context)._context)


cdef int _jacobian_provider_error(void* context,int status) noexcept with gil:
    provider=<_JacobianDifferentiation><object>context
    error=provider._real._error if provider._adaptive else provider._complex._error
    if error is not None:
        provider._error=error
        if provider._adaptive:provider._real._error=None
        else:provider._complex._error=None
    elif status!=ESTIMATE_CALLBACK_FAILURE:
        provider._error=DifferentiationError(provider._estimate())
    return provider._report.callback_code if status==ESTIMATE_CALLBACK_FAILURE else -1000-status


cdef int _jacobian_provider(const double* x,size_t n,size_t m,double* out,void* context) noexcept nogil:
    cdef int status=(<_JacobianDifferentiation><object>context)._compute(x)
    if status==ESTIMATE_OK:
        memcpy(out,(<_JacobianDifferentiation><object>context)._storage.data(),n*m*sizeof(double))
        return 0
    return _jacobian_provider_error(context,status)


cdef class _JacobianDifferentiation:
    def __init__(self):raise TypeError('Choose ComplexStepJacobian or AdaptiveJacobian.')

    cdef void _configure(self,object function,object input_dimension,object output_dimension,
            bint adaptive,object step,object atol,object rtol,object max_steps,object memory_bytes) except *:
        cdef size_t i
        cdef ComplexVectorCallback complex_source
        if (self._function!=NULL or self._constructing or self._storage!=NULL or
                self._steps!=NULL or self._real is not None or self._complex is not None):
            raise RuntimeError('Jacobian providers cannot be reinitialized.')
        self._constructing=True
        try:
            n,m=_shape(input_dimension,output_dimension)
            limit=_checked_integer(memory_bytes,'memory_bytes')
            self._atol,self._rtol,self._levels=_settings(atol,rtol,max_steps)
            work=n+(int(self._levels)+5)*m if adaptive else 2*n+2*m
            count=2*n*m+n+work
            step_count=0 if step is None else n
            total=(count+step_count)*sizeof(double)
            if count>OwnedDoubleBuffer.max_size() or total>limit:
                raise ValueError('Jacobian provider exceeds memory_bytes or native capacity.')
            self._n,self._m,self._output_size=n,m,n*m
            self._jacobian,self._adaptive=True,adaptive
            if step is not None:
                self._steps=new OwnedDoubleBuffer(n)
                if not hasattr(step,'__len__') or isinstance(step,(str,bytes,bytearray)):
                    value=_number(step,'step',True)
                    for i in range(self._n):self._steps.set(i,value)
                else:
                    if len(step)!=n:raise ValueError('Step vector length must match input_dimension.')
                    for i in range(self._n):self._steps.set(i,_number(step[i],'step',True))
            if adaptive:
                self._real=_private_vector(function,self._n,self._m)
                self._native=self._real._native
            else:
                if isinstance(function,ComplexVectorCallback):
                    complex_source=function
                    if complex_source._function==NULL or complex_source._active or complex_source._constructing:
                        raise RuntimeError('Complex callback is uninitialized or active.')
                    if complex_source._n!=self._n or complex_source._m!=self._m:
                        raise ValueError('Complex callback dimensions must match the estimator.')
                    self._complex=make_complex_vector_callback(complex_source._function,self._n,self._m,complex_source._context,complex_source) if complex_source._native else ComplexVectorCallback(complex_source._python_function,n,m)
                else:self._complex=ComplexVectorCallback(function,n,m)
                self._native=self._complex._native
            self._storage=new OwnedDoubleBuffer(count)
            self._memory_bytes=total
            self._report.status=-1
            self._context=<void*>self
            self._function=_jacobian_provider
        finally:
            self._constructing=False

    def __dealloc__(self):
        if self._storage!=NULL:del self._storage
        if self._steps!=NULL:del self._steps

    cdef int _compute(self,const double* x) noexcept nogil:
        cdef double* steps=NULL
        cdef double* out=self._storage.data()
        cdef double* errors=out+self._n*self._m
        cdef double* point=errors+self._n*self._m
        cdef double* work=point+self._n
        if self._steps!=NULL:steps=self._steps.data()
        if self._adaptive:
            return adaptive_jacobian(x,self._n,self._m,steps,self._atol,self._rtol,self._levels,
                out,errors,work,_vector_real,<void*>self._real,self._report)
        return complex_step_jacobian(x,self._n,self._m,steps,out,work,
            _vector_complex,<void*>self._complex,self._report)

    cdef object _estimate(self):
        cdef size_t i,j
        cdef double* out=self._storage.data()
        cdef double* errors=out+self._n*self._m
        if self._report.status<0:raise RuntimeError('No Jacobian has been evaluated.')
        value=tuple([tuple([out[i+j*self._m] for j in range(self._n)]) for i in range(self._m)])
        error=tuple([tuple([errors[i+j*self._m] for j in range(self._n)]) for i in range(self._m)]) if self._adaptive else None
        return NumericalJacobianEstimate(value,error,self._report.calls,_status(self._report.status),self._report.callback_code)

    def evaluate(self,x,*,workspace_bytes=None):
        """Return a copied Jacobian/error estimate, including nonconvergence.

        Scratch is allocated at construction, bounded by memory_bytes. The
        optional workspace_bytes argument can impose a stricter call-time cap.
        """
        cdef double* point
        self._enter()
        try:
            if workspace_bytes is not None and self._memory_bytes>_checked_integer(workspace_bytes,'workspace_bytes'):
                raise ValueError('Estimator storage exceeds workspace_bytes.')
            point=self._storage.data()+2*self._n*self._m
            _pack_point(x,point,self._n)
            with nogil:self._compute(point)
            estimate=self._estimate()
            error=self._real._error if self._adaptive else self._complex._error
            if error is not None:
                if self._adaptive:self._real._error=None
                else:self._complex._error=None
                raise error
            if self._report.status==ESTIMATE_CALLBACK_FAILURE:
                raise NativeCallbackError(self._report.callback_code,estimate,'function')
            return estimate
        finally:
            self._release()

    def __call__(self,x):
        estimate=self.evaluate(x)
        if estimate.status!='ok':raise DifferentiationError(estimate)
        return estimate.jacobian

    @property
    def last_estimate(self):
        if self._function==NULL or self._constructing:raise RuntimeError('Estimator is not initialized.')
        if self._active:raise RuntimeError('This Jacobian provider is already active.')
        return self._estimate()
    @property
    def step(self):
        return None if self._steps==NULL else tuple([self._steps.get(i) for i in range(self._n)])
    @property
    def memory_bytes(self):return int(self._memory_bytes)
    @property
    def max_steps(self):return int(self._levels) if self._adaptive else None


cdef class ComplexStepJacobian:
    """Rectangular first derivatives from a holomorphic complex residual.

    n complex calls yield an m-by-n Jacobian. See ComplexStepDerivative for
    analyticity restrictions and default step. Each step can also be a length-n
    sequence. Reuses private scratch; one provider cannot be called concurrently.
    """
    def __init__(self,function,input_dimension,output_dimension,*,step=None,memory_bytes=64*1024*1024):
        self._configure(function,input_dimension,output_dimension,False,step,0.,0.,2,memory_bytes)


cdef class AdaptiveJacobian:
    """Rectangular Jacobian with bounded Richardson refinement per input axis.

    Uses one base residual plus two residual calls per refinement per axis.
    Every component must meet its own atol+rtol*abs(estimate) threshold.
    Nonconvergence returns best estimates through evaluate(); solver use raises.
    """
    def __init__(self,function,input_dimension,output_dimension,*,step=None,
                 atol=1e-10,rtol=1e-8,max_steps=10,memory_bytes=64*1024*1024):
        self._configure(function,input_dimension,output_dimension,True,step,atol,rtol,max_steps,memory_bytes)


cdef int _hessian_provider_error(void* context,int status) noexcept with gil:
    provider=<AdaptiveHessian><object>context
    if provider._real._error is not None:
        provider._error=provider._real._error
        provider._real._error=None
    elif status!=ESTIMATE_CALLBACK_FAILURE:
        provider._error=DifferentiationError(provider._estimate())
    return provider._report.callback_code if status==ESTIMATE_CALLBACK_FAILURE else -1000-status


cdef int _hessian_provider(const double* x,size_t n,size_t m,double* out,void* context) noexcept nogil:
    cdef int status=(<AdaptiveHessian><object>context)._compute(x)
    if status==ESTIMATE_OK:
        memcpy(out,(<AdaptiveHessian><object>context)._storage.data(),n*n*m*sizeof(double))
        return 0
    return _hessian_provider_error(context,status)


cdef class AdaptiveHessian:
    """Residual Hessians of shape (m,n,n), including mixed partials.

    Central differences on actual rounded nodes are Richardson-extrapolated
    together as a tensor. Each level uses 2*n*n residual calls, plus one base
    call per evaluation. step=None starts at .1*max(1,abs(x[j])); an explicit
    positive scalar or length-n sequence supplies absolute initial steps.
    Every entry must meet atol+rtol*abs(estimate). Errors are estimates, not
    bounds: smooth deterministic residuals throughout the stencil are required.
    Scratch, including the O(max_steps*m*n*n) table, is bounded by memory_bytes.
    evaluate() retains the best estimate on nonconvergence; solver use raises.
    """
    def __init__(self,function,input_dimension,output_dimension,*,step=None,
                 atol=1e-10,rtol=1e-8,max_steps=10,memory_bytes=64*1024*1024):
        cdef size_t i
        if (self._function!=NULL or self._constructing or self._storage!=NULL or
                self._steps!=NULL or self._real is not None):
            raise RuntimeError('Hessian providers cannot be reinitialized.')
        self._constructing=True
        try:
            n,m=_shape(input_dimension,output_dimension)
            limit=_checked_integer(memory_bytes,'memory_bytes')
            self._atol,self._rtol,self._levels=_settings(atol,rtol,max_steps)
            tensor=n*n*m
            work=2*n*m+8*n+2*m
            count=(int(self._levels)+5)*tensor+4*n+m+work
            total=(count+(0 if step is None else n))*sizeof(double)
            if count>OwnedDoubleBuffer.max_size() or total>limit:
                raise ValueError('Hessian provider exceeds memory_bytes or native capacity.')
            self._n,self._m,self._tensor_size=n,m,tensor
            self._fixed_work,self._memory_bytes=work,total
            if step is not None:
                self._steps=new OwnedDoubleBuffer(n)
                if not hasattr(step,'__len__') or isinstance(step,(str,bytes,bytearray)):
                    value=_number(step,'step',True)
                    for i in range(self._n):self._steps.set(i,value)
                else:
                    if len(step)!=n:raise ValueError('Step vector length must match input_dimension.')
                    for i in range(self._n):self._steps.set(i,_number(step[i],'step',True))
            self._real=_private_vector(function,self._n,self._m)
            self._storage=new OwnedDoubleBuffer(count)
            self._report.status=-1
            self._native=self._real._native
            self._context=<void*>self
            self._function=_hessian_provider
        finally:
            self._constructing=False

    def __dealloc__(self):
        if self._storage!=NULL:del self._storage
        if self._steps!=NULL:del self._steps

    cdef int _compute(self,const double* x) noexcept nogil:
        cdef size_t n=self._n,m=self._m,t=self._tensor_size,i,j,k,stalled=0
        cdef double* out=self._storage.data()
        cdef double* errors=out+t
        # The n doubles after errors hold the packed public input.
        cdef double* steps=errors+t+n
        cdef double* nodes=steps+n
        cdef double* base=nodes+2*n
        cdef double* raw=base+m
        cdef double* table=raw+t
        cdef double* diagonal=table+self._levels*t
        cdef double* previous=diagonal+t
        cdef double* work=previous+t
        cdef double xp,xm,a,b
        cdef int status,improved
        cdef bint converged,deteriorated
        cdef RectangularHessianConfig config
        self._report.calls,self._report.status,self._report.callback_code=0,ESTIMATE_OK,0
        for i in range(t):out[i]=NAN;errors[i]=INFINITY
        for i in range((self._levels+2)*t):table[i]=0.
        for j in range(n):
            if not isfinite(x[j]):
                return diff_failure(self._report,ESTIMATE_NONFINITE,out,errors,t)
            steps[j]=self._steps.data()[j] if self._steps!=NULL else .1*fmax(1.,fabs(x[j]))
        for i in range(m):base[i]=NAN
        self._report.calls+=1
        status=_vector_real(x,n,m,base,<void*>self._real)
        if status:
            self._report.callback_code=status
            return diff_failure(self._report,ESTIMATE_CALLBACK_FAILURE,out,errors,t)
        for i in range(m):
            if not isfinite(base[i]):
                return diff_failure(self._report,ESTIMATE_NONFINITE,out,errors,t)
        config.input_dimension,config.output_dimension=n,m
        config.matrix_size,config.tensor_size=n*m,t
        config.work,config.scheme,config.steps=self._fixed_work,2,steps
        for k in range(self._levels):
            for j in range(n):
                xp,xm=x[j]+steps[j],x[j]-steps[j]
                a,b=xp-x[j],xm-x[j]
                if not isfinite(xp) or not isfinite(xm) or not isfinite(a) or not isfinite(b):
                    return diff_failure(self._report,ESTIMATE_NONFINITE,out,errors,t)
                if a==0. or b==0. or xp==xm:
                    if not k:return diff_failure(self._report,ESTIMATE_BREAKDOWN,out,errors,t)
                    self._report.status=ESTIMATE_ROUNDOFF
                    return ESTIMATE_ROUNDOFF
                if k and xp==nodes[2*j] and xm==nodes[2*j+1]:
                    self._report.status=ESTIMATE_ROUNDOFF
                    return ESTIMATE_ROUNDOFF
                nodes[2*j],nodes[2*j+1]=xp,xm
            status=compute_rectangular_hessian(&config,x,base,raw,t,work,self._fixed_work,
                _vector_real,<void*>self._real,&self._report.calls,&self._report.callback_code)
            if status:return diff_failure(self._report,status,out,errors,t)
            status=richardson_update(raw,t,k,table,diagonal,previous,out,errors,&improved)
            if status:return diff_failure(self._report,status,out,errors,t)
            if k:
                stalled=0 if improved else stalled+1
                converged,deteriorated=True,True
                for i in range(t):
                    if not errors[i]<=self._atol+self._rtol*fabs(out[i]):converged=False
                    if not fabs(diagonal[i]-previous[i])>2.*errors[i]:deteriorated=False
                if converged:return ESTIMATE_OK
                if k>=4 and stalled>=2 and deteriorated:
                    self._report.status=ESTIMATE_ROUNDOFF
                    return ESTIMATE_ROUNDOFF
            for j in range(n):steps[j]*=.5
        self._report.status=ESTIMATE_MAX_STEPS
        return ESTIMATE_MAX_STEPS

    cdef object _estimate(self):
        cdef size_t i,j,k
        cdef double* out=self._storage.data()
        cdef double* errors=out+self._tensor_size
        if self._report.status<0:raise RuntimeError('No Hessian has been evaluated.')
        value=tuple([tuple([tuple([out[i+self._m*(j+self._n*k)] for k in range(self._n)]) for j in range(self._n)]) for i in range(self._m)])
        error=tuple([tuple([tuple([errors[i+self._m*(j+self._n*k)] for k in range(self._n)]) for j in range(self._n)]) for i in range(self._m)])
        return NumericalHessianEstimate(value,error,self._report.calls,_status(self._report.status),self._report.callback_code)

    def evaluate(self,x,*,workspace_bytes=None):
        cdef double* point
        self._enter()
        try:
            if workspace_bytes is not None and self._memory_bytes>_checked_integer(workspace_bytes,'workspace_bytes'):
                raise ValueError('Estimator storage exceeds workspace_bytes.')
            point=self._storage.data()+2*self._tensor_size
            _pack_point(x,point,self._n)
            with nogil:self._compute(point)
            estimate=self._estimate()
            if self._real._error is not None:
                error=self._real._error;self._real._error=None
                raise error
            if self._report.status==ESTIMATE_CALLBACK_FAILURE:
                raise NativeCallbackError(self._report.callback_code,estimate,'function')
            return estimate
        finally:
            self._release()

    def __call__(self,x):
        estimate=self.evaluate(x)
        if estimate.status!='ok':raise DifferentiationError(estimate)
        return estimate.hessian

    @property
    def last_estimate(self):
        if self._function==NULL or self._constructing:raise RuntimeError('Estimator is not initialized.')
        if self._active:raise RuntimeError('This Hessian provider is already active.')
        return self._estimate()
    @property
    def step(self):
        return None if self._steps==NULL else tuple([self._steps.get(i) for i in range(self._n)])
    @property
    def memory_bytes(self):return int(self._memory_bytes)
    @property
    def max_steps(self):return int(self._levels)
