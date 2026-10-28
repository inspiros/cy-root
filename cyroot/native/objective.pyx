# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Exact scalar least-squares objective with reusable native callback storage."""
from libc.limits cimport INT_MAX
from libc.math cimport NAN, isfinite
from libc.string cimport memcpy
from numbers import Complex, Real
from cpython.buffer cimport PyObject_CheckBuffer
from .buffers cimport VectorView
from ._buffers cimport DoubleBuffer
from .higher_derivatives import _checked_integer
from .scalar import NativeCallbackError


cdef void _fill(double* out, size_t count) noexcept nogil:
    cdef size_t i
    for i in range(count):out[i] = NAN


cdef bint _finite(const double* values, size_t count) noexcept nogil:
    cdef size_t i
    for i in range(count):
        if not isfinite(values[i]):return False
    return True


cdef void _pack(object point, double* out, size_t n) except *:
    cdef DoubleBuffer guard
    cdef VectorView view
    cdef size_t i
    if isinstance(point, (str, bytes, bytearray)):
        raise TypeError('Expected a numeric point vector.')
    if PyObject_CheckBuffer(point):
        guard = DoubleBuffer(point)
        view = guard.vector_view()
        if view.size() != n:raise ValueError('Point length must match input_dimension.')
        for i in range(n):out[i] = view.get(i)
    else:
        if len(point) != n:raise ValueError('Point length must match input_dimension.')
        for i in range(n):
            value=point[i]
            if isinstance(value,Complex) and not isinstance(value,Real):
                raise TypeError('Point coordinates must be real scalars.')
            out[i] = value


cdef class LeastSquaresObjective:
    """phi(x)=sum(F_i(x)**2)/2, gradient=J.T F, exact Hessian=J.T J+sum F_i H_i.

    Derivative providers are explicit; no Gauss-Newton substitution or numerical
    differentiation is selected implicitly. Each operation reuses private
    native storage. A state and its callback adapters cannot be used reentrantly.
    """
    def __init__(self, function, input_dimension, output_dimension=None, *,
                 jacobian=None, hessian=None, workspace_bytes=64*1024*1024, scalar=False):
        if (self._constructing or self._initialized or self._storage != NULL or
                self._function is not None or self._rect_function is not None or
                self._jacobian is not None or self._rect_jacobian is not None or
                self._hessian is not None or self._rect_hessian is not None or
                self._scalar_function is not None or self._scalar_jacobian is not None or
                self._scalar_hessian is not None):
            raise RuntimeError('Objective adapters cannot be reinitialized.')
        self._constructing = True
        reserved = False
        try:
            # Attach and reserve original native adapters before any user
            # dimension/option conversion can execute arbitrary Python.
            if isinstance(function, ScalarCallback):self._scalar_function = function
            if isinstance(jacobian, ScalarCallback):self._scalar_jacobian = jacobian
            if isinstance(hessian, ScalarCallback):self._scalar_hessian = hessian
            if isinstance(function, VectorCallback):self._function = function
            elif isinstance(function, RectangularCallback):self._rect_function = function
            if isinstance(jacobian, VectorCallback):self._jacobian = jacobian
            elif isinstance(jacobian, RectangularCallback):self._rect_jacobian = jacobian
            if isinstance(hessian, HessianCallback):self._hessian = hessian
            elif isinstance(hessian, RectangularHessianCallback):self._rect_hessian = hessian
            self._reserve()
            reserved = True
            if type(scalar) is not bool:raise TypeError('scalar must be bool.')
            self._scalar = scalar or self._scalar_function is not None
            if not self._scalar and (self._scalar_jacobian is not None or self._scalar_hessian is not None):
                raise ValueError('Scalar derivative adapters require a scalar residual.')
            if hessian is not None and jacobian is None:
                raise ValueError('An explicit Jacobian provider is required with a residual Hessian provider.')
            n = _checked_integer(input_dimension, 'input_dimension')
            m = n if output_dimension is None else _checked_integer(output_dimension, 'output_dimension')
            limit = _checked_integer(workspace_bytes, 'workspace_bytes')
            if n == 0 or m == 0 or n > INT_MAX or m > INT_MAX:
                raise ValueError('Positive dimensions must fit native capacities and C int.')
            if self._scalar and (n != 1 or m != 1):
                raise ValueError('Scalar providers require input/output dimension one.')
            if self._scalar:
                if (self._function is not None or self._rect_function is not None or
                    self._jacobian is not None or self._rect_jacobian is not None or
                    self._hessian is not None or self._rect_hessian is not None):
                    raise TypeError('Scalar mode requires scalar callables or ScalarCallback providers.')
                if self._scalar_function is None:
                    self._scalar_function=ScalarCallback(function)
                    self._scalar_function._active=True
                if jacobian is not None and self._scalar_jacobian is None:
                    self._scalar_jacobian=ScalarCallback(jacobian)
                    self._scalar_jacobian._active=True
                if hessian is not None and self._scalar_hessian is None:
                    self._scalar_hessian=ScalarCallback(hessian)
                    self._scalar_hessian._active=True
            matrix = n*m
            tensor = n*n*m if hessian is not None else 0
            capacity = max(m, 1, n if jacobian is not None else 0,
                           n*n if hessian is not None else 0)
            count = n+m+capacity+(matrix if jacobian is not None else 0)+(tensor if hessian is not None else 0)
            if count > OwnedDoubleBuffer.max_size() or count*sizeof(double) > limit:
                raise ValueError('Objective storage exceeds workspace_bytes or native capacities.')
            if self._scalar:pass
            elif isinstance(function, VectorCallback):
                self._function = function
                if self._function._function == NULL or self._function._constructing:
                    raise RuntimeError('Residual callback is not initialized.')
                if self._function._jacobian or self._function._dimension != n or m != n:
                    raise ValueError('Residual callback kind and dimensions must match.')
            elif isinstance(function, RectangularCallback):
                self._rect_function = function
                if self._rect_function._function == NULL or self._rect_function._constructing:
                    raise RuntimeError('Residual callback is not initialized.')
                if self._rect_function._jacobian or self._rect_function._n != n or self._rect_function._m != m:
                    raise ValueError('Residual callback kind and dimensions must match.')
            else:
                if not callable(function):raise TypeError('function must be a residual callable or callback.')
                self._rect_function = RectangularCallback(function, n, m)
                self._rect_function._active = True
            if self._scalar:pass
            elif isinstance(jacobian, VectorCallback):
                self._jacobian = jacobian
                if self._jacobian._function == NULL or self._jacobian._constructing:
                    raise RuntimeError('Jacobian callback is not initialized.')
                if not self._jacobian._jacobian or self._jacobian._dimension != n or m != n:
                    raise ValueError('Jacobian callback kind and dimensions must match.')
            elif isinstance(jacobian, RectangularCallback):
                self._rect_jacobian = jacobian
                if self._rect_jacobian._function == NULL or self._rect_jacobian._constructing:
                    raise RuntimeError('Jacobian callback is not initialized.')
                if not self._rect_jacobian._jacobian or self._rect_jacobian._n != n or self._rect_jacobian._m != m:
                    raise ValueError('Jacobian callback kind and dimensions must match.')
            elif jacobian is not None:
                if not callable(jacobian):raise TypeError('jacobian must be a Jacobian callable or callback.')
                self._rect_jacobian = RectangularCallback(jacobian, n, m, jacobian=True)
                self._rect_jacobian._active = True
            if self._scalar:pass
            elif isinstance(hessian, HessianCallback):
                self._hessian = hessian
                if self._hessian._function == NULL or self._hessian._constructing:
                    raise RuntimeError('Hessian callback is not initialized.')
                if self._hessian._dimension != n or m != n:
                    raise ValueError('Hessian callback dimensions must match.')
            elif isinstance(hessian, RectangularHessianCallback):
                self._rect_hessian = hessian
                if self._rect_hessian._function == NULL or self._rect_hessian._constructing:
                    raise RuntimeError('Hessian callback is not initialized.')
                if self._rect_hessian._n != n or self._rect_hessian._m != m:
                    raise ValueError('Hessian callback dimensions must match.')
            elif hessian is not None:
                if not callable(hessian):raise TypeError('hessian must be a Hessian callable or callback.')
                self._rect_hessian = RectangularHessianCallback(hessian, n, m)
                self._rect_hessian._active = True
            self._n, self._m, self._matrix, self._tensor = n, m, matrix, tensor
            self._capacity, self._memory_bytes = capacity, count*sizeof(double)
            self._storage = new OwnedDoubleBuffer(count)
            self._x = self._storage.data()
            self._f = self._x+self._n
            self._result = self._f+self._m
            if jacobian is not None:self._j = self._result+self._capacity
            if hessian is not None:self._h = self._result+self._capacity+(self._matrix if jacobian is not None else 0)
            self._initialized = True
        finally:
            if reserved:self._leave()
            self._constructing = False

    @classmethod
    def from_scalar(cls,function,*,derivative=None,second_derivative=None,
                    workspace_bytes=64*1024*1024):
        """Adapt f(x), f'(x), f''(x) directly to one-dimensional native storage.

        Objective evaluation still uses point vectors. Original ScalarCallback
        providers remain reserved/pinned; no Python vector bridge is introduced.
        """
        return cls(function,1,jacobian=derivative,hessian=second_derivative,
                   scalar=True,workspace_bytes=workspace_bytes)

    def __dealloc__(self):
        if self._storage != NULL:del self._storage

    cdef void _ready(self) except *:
        if self._constructing or not self._initialized:
            raise RuntimeError('Objective adapter is not initialized.')
        if self._busy:raise RuntimeError('This objective adapter is already active.')

    cdef void _enter(self) except *:
        self._ready()
        self._reserve()

    cdef void _reserve(self) except *:
        if ((self._scalar_function is not None and self._scalar_function._function == NULL) or
            (self._scalar_jacobian is not None and self._scalar_jacobian._function == NULL) or
            (self._scalar_hessian is not None and self._scalar_hessian._function == NULL)):
            raise RuntimeError('Scalar callback adapter is not initialized.')
        if ((self._scalar_function is not None and self._scalar_function._active) or
            (self._scalar_jacobian is not None and self._scalar_jacobian._active) or
            (self._scalar_hessian is not None and self._scalar_hessian._active)):
            raise RuntimeError('This scalar callback adapter is already active.')
        if ((self._function is not None and (self._function._function == NULL or self._function._constructing)) or
                (self._rect_function is not None and (self._rect_function._function == NULL or self._rect_function._constructing)) or
                (self._jacobian is not None and (self._jacobian._function == NULL or self._jacobian._constructing)) or
                (self._rect_jacobian is not None and (self._rect_jacobian._function == NULL or self._rect_jacobian._constructing)) or
                (self._hessian is not None and (self._hessian._function == NULL or self._hessian._constructing)) or
                (self._rect_hessian is not None and (self._rect_hessian._function == NULL or self._rect_hessian._constructing))):
            raise RuntimeError('Callback adapter is not initialized.')
        if ((self._function is not None and self._function._active) or
                (self._rect_function is not None and self._rect_function._active) or
                (self._jacobian is not None and self._jacobian._active) or
                (self._rect_jacobian is not None and self._rect_jacobian._active) or
                (self._hessian is not None and self._hessian._active) or
                (self._rect_hessian is not None and self._rect_hessian._active)):
            raise RuntimeError('This callback adapter is already active.')
        self._busy = True
        if self._scalar_function is not None:self._scalar_function._active=True
        if self._scalar_jacobian is not None:self._scalar_jacobian._active=True
        if self._scalar_hessian is not None:self._scalar_hessian._active=True
        if self._function is not None:self._function._active = True
        if self._rect_function is not None:self._rect_function._active = True
        if self._jacobian is not None:self._jacobian._active = True
        if self._rect_jacobian is not None:self._rect_jacobian._active = True
        if self._hessian is not None:self._hessian._active = True
        if self._rect_hessian is not None:self._rect_hessian._active = True
        if self._scalar_function is not None:self._scalar_function._error=None
        if self._scalar_jacobian is not None:self._scalar_jacobian._error=None
        if self._scalar_hessian is not None:self._scalar_hessian._error=None
        if self._function is not None:self._function._error = None
        if self._rect_function is not None:self._rect_function._error = None
        if self._jacobian is not None:self._jacobian._error = None
        if self._rect_jacobian is not None:self._rect_jacobian._error = None
        if self._hessian is not None:self._hessian._error = None
        if self._rect_hessian is not None:self._rect_hessian._error = None

    cdef void _leave(self) noexcept:
        if self._scalar_function is not None:self._scalar_function._active=False
        if self._scalar_jacobian is not None:self._scalar_jacobian._active=False
        if self._scalar_hessian is not None:self._scalar_hessian._active=False
        if self._function is not None:self._function._active = False
        if self._rect_function is not None:self._rect_function._active = False
        if self._jacobian is not None:self._jacobian._active = False
        if self._rect_jacobian is not None:self._rect_jacobian._active = False
        if self._hessian is not None:self._hessian._active = False
        if self._rect_hessian is not None:self._rect_hessian._active = False
        self._busy = False

    cdef int _sample(self, int stage, const double* x, double* out) noexcept nogil:
        cdef int code
        cdef size_t count = self._m if stage == 1 else self._matrix if stage == 2 else self._tensor
        _fill(out, count)
        if self._scalar:
            if stage == 1:
                code=self._scalar_function._function(x[0],out,self._scalar_function._context)
            elif stage == 2:
                code=self._scalar_jacobian._function(x[0],out,self._scalar_jacobian._context)
            else:code=self._scalar_hessian._function(x[0],out,self._scalar_hessian._context)
        elif stage == 1:
            if self._function is not None:
                code = self._function._function(x, self._n, out, self._function._context)
            else:
                code = self._rect_function._function(x, self._n, self._m, out, self._rect_function._context)
        elif stage == 2:
            if self._jacobian is not None:
                code = self._jacobian._function(x, self._n, out, self._jacobian._context)
            else:
                code = self._rect_jacobian._function(x, self._n, self._m, out, self._rect_jacobian._context)
        else:
            if self._hessian is not None:
                code = self._hessian._function(x, self._n, out, self._hessian._context)
            else:
                code = self._rect_hessian._function(x, self._n, self._m, out, self._rect_hessian._context)
        if code != 0:
            self._callback_code, self._callback_stage = code, stage
            return 3
        if not _finite(out, count):return 1
        return 0

    cdef int evaluate_raw(self, const double* x, int order, double* out,
                          size_t capacity) noexcept nogil:
        cdef size_t needed
        cdef int status
        if not self._initialized or self._constructing or not self._busy:return 2
        self._callback_code, self._callback_stage = 0, 0
        if order < -1 or order > 2 or x == NULL or out == NULL:return 2
        needed = self._m if order == -1 else 1 if order == 0 else self._n if order == 1 else self._n*self._n
        if capacity < needed:return 2
        if order >= 1 and self._j == NULL:return 2
        if order == 2 and self._h == NULL:return 2
        if not _finite(x, self._n):return 1
        status = self._sample(1, x, self._f)
        if status != 0:return status
        if order == -1:
            memcpy(out, self._f, self._m*sizeof(double))
            return 0
        if order == 0:
            status = objective_value(self._f, self._m, self._result)
        else:
            status = self._sample(2, x, self._j)
            if status != 0:return status
            if order == 1:
                status = objective_gradient(self._f, self._j, self._n, self._m, self._result)
            else:
                status = self._sample(3, x, self._h)
                if status != 0:return status
                status = objective_hessian(self._f, self._j, self._h, self._n, self._m, self._result)
        if status == 0 and out != self._result:
            memcpy(out, self._result, needed*sizeof(double))
        return status

    cdef void _raise_callback_error(self, object result) except *:
        cdef object error = None
        if self._scalar_function is not None and self._scalar_function._error is not None:
            error,self._scalar_function._error=self._scalar_function._error,None
        elif self._scalar_jacobian is not None and self._scalar_jacobian._error is not None:
            error,self._scalar_jacobian._error=self._scalar_jacobian._error,None
        elif self._scalar_hessian is not None and self._scalar_hessian._error is not None:
            error,self._scalar_hessian._error=self._scalar_hessian._error,None
        elif self._function is not None and self._function._error is not None:
            error, self._function._error = self._function._error, None
        elif self._rect_function is not None and self._rect_function._error is not None:
            error, self._rect_function._error = self._rect_function._error, None
        elif self._jacobian is not None and self._jacobian._error is not None:
            error, self._jacobian._error = self._jacobian._error, None
        elif self._rect_jacobian is not None and self._rect_jacobian._error is not None:
            error, self._rect_jacobian._error = self._rect_jacobian._error, None
        elif self._hessian is not None and self._hessian._error is not None:
            error, self._hessian._error = self._hessian._error, None
        elif self._rect_hessian is not None and self._rect_hessian._error is not None:
            error, self._rect_hessian._error = self._rect_hessian._error, None
        if error is not None:raise error
        if self._callback_code != 0:
            failure=NativeCallbackError(self._callback_code, result,
                'function' if self._callback_stage == 1 else 'jacobian' if self._callback_stage == 2 else 'hessian')
            # A public facade may normalize this solver-produced snapshot.
            # A NativeCallbackError raised by Python user code must pass through
            # unchanged, even though it has the same exception class.
            failure._cyroot_objective=self
            raise failure

    cdef object _evaluate(self, object point, int order):
        cdef int status
        cdef size_t i, j, needed
        cdef bint copy_point=order == -2
        if copy_point:order=-1
        self._enter()
        try:
            if order >= 1 and self._j == NULL:
                raise ValueError('An explicit Jacobian provider is required.')
            if order == 2 and self._h == NULL:
                raise ValueError('An explicit residual Hessian provider is required for the exact objective Hessian.')
            _pack(point, self._x, self._n)
            needed = self._m if order == -1 else 1 if order == 0 else self._n if order == 1 else self._n*self._n
            _fill(self._result, needed)
            # Internal export uses the staging buffer itself. External raw
            # callers keep their output disjoint for transactional writes.
            with nogil:status = self.evaluate_raw(self._x, order, self._result, needed)
            if order == 0:result = self._result[0]
            elif order == 2:
                result = tuple([tuple([self._result[i+self._n*j] for j in range(self._n)]) for i in range(self._n)])
            else:result = tuple([self._result[i] for i in range(needed)])
            self._raise_callback_error(result)
            if status == 1:raise FloatingPointError('Objective input, callback output or final result is nonfinite.')
            if status != 0:raise RuntimeError('Invalid native objective evaluation.')
            return (tuple([self._x[i] for i in range(self._n)]),result) if copy_point else result
        finally:
            self._leave()

    def value(self, point):return self._evaluate(point, 0)
    def __call__(self, point):return self._evaluate(point, 0)
    def gradient(self, point):return self._evaluate(point, 1)
    def hessian(self, point):return self._evaluate(point, 2)
    def residual(self, point):return self._evaluate(point, -1)
    def _initial_state(self, point):return self._evaluate(point, -2)

    @property
    def input_dimension(self):self._ready();return self._n
    @property
    def output_dimension(self):self._ready();return self._m
    @property
    def memory_bytes(self):self._ready();return self._memory_bytes
    @property
    def callback_code(self):self._ready();return self._callback_code
    @property
    def callback_stage(self):
        self._ready()
        return ('function' if self._callback_stage == 1 else 'jacobian' if self._callback_stage == 2 else
                'hessian' if self._callback_stage == 3 else None)
