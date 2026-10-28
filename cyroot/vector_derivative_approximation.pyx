# distutils: language=c++
# cython: cdivision=True, initializedcheck=False, boundscheck=False, profile=False
"""NumPy-facing compatibility boundary over native derivative plans."""
from typing import Callable, Optional, Union
import numpy as np
cimport numpy as np
import cython
from libc.stddef cimport size_t
from ._types import VectorLike, ArrayLike
from .fptr cimport NdArrayFPtr, PyNdArrayFPtr
from .native.rectangular cimport RectangularCallback, RectangularFunction, make_rectangular_callback
from .native.vector cimport VectorCallback, VectorFunction
from .native.derivatives import RectangularJacobianPlan
from .native.hessians import RectangularHessianPlan
from .native.higher_mixed import TensorDifferencePlan
from .native.tensor_callbacks cimport HessianCallback, RectangularHessianCallback
from .scalar_derivative_approximation import _check_finite_difference_args
from .utils._function_registering import register

__all__ = ['VectorDerivativeApproximation','GeneralizedFiniteDifference',
           'generalized_finite_difference']
_MAX_TENSOR_RANK = 32
_MAX_STENCIL_TERMS = 1_000_000
_MAX_WORK_BYTES = 256*1024*1024


def _point_array(x):
    x = np.array(x, dtype=np.float64, copy=True)
    if x.ndim != 1 or x.size == 0:
        raise ValueError('x must be a non-empty one-dimensional array.')
    if not np.all(np.isfinite(x)):
        raise ValueError('x must contain only finite values.')
    return x


def _step_array(h, size):
    # Also validate attributes changed after constructing a derivative wrapper.
    h, _, _ = _check_finite_difference_args(h, 1, 0)
    if isinstance(h, float):
        return np.full(size, h, dtype=np.float64)
    if len(h) != size:
        raise ValueError(f'h must be scalar or have length {size}; got {len(h)}.')
    return np.array(h, dtype=np.float64, copy=True)


def _value_array(value, size=None):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or value.size == 0:
        raise ValueError('F(x) and F_x must be non-empty one-dimensional arrays.')
    if size is not None and value.size != size:
        raise ValueError(f'F changed its output length from {size} to {value.size}.')
    if not np.all(np.isfinite(value)):
        raise ValueError('F(x) and F_x must contain only finite values.')
    # Callbacks may reuse their own output array on the next evaluation.
    return value.copy()


cdef class _ArrayResidual:
    """Legacy ndarray callbacks are converted only at their Python boundary."""
    cdef NdArrayFPtr function
    cdef Py_ssize_t output_dimension

    def __call__(self, point):
        return _value_array(self.function.eval(np.array(point,dtype=np.float64)),
                            self.output_dimension)


cdef struct _SquareBridge:
    VectorFunction function
    void* context


cdef int _square_residual(const double* x, size_t n, size_t m, double* out,
                         void* context) noexcept nogil:
    cdef _SquareBridge* bridge = <_SquareBridge*>context
    return bridge.function(x,n,out,bridge.context)


cdef np.ndarray _difference(object function, object x, object cached,
                            object step, object order, object kind,
                            GeneralizedFiniteDifference owner=None):
    cdef RectangularCallback rectangular = None
    cdef VectorCallback square = None
    cdef RectangularCallback proxy = None
    cdef NdArrayFPtr legacy = None
    cdef _ArrayResidual array_bridge
    cdef _SquareBridge square_bridge
    cdef bint reserved = False
    cdef np.ndarray point, base, steps, result
    # Reserve explicit adapters before arbitrary input/configuration conversions.
    if isinstance(function,RectangularCallback):
        rectangular = function
        if rectangular._function == NULL or rectangular._constructing:
            raise RuntimeError('Residual callback adapter is not initialized.')
        if rectangular._jacobian:raise ValueError('F requires a residual-kind callback.')
        rectangular._enter()
        reserved = True
    elif isinstance(function,VectorCallback):
        square = function
        if square._function == NULL or square._constructing:
            raise RuntimeError('Residual callback adapter is not initialized.')
        if square._jacobian:raise ValueError('F requires a residual-kind callback.')
        if square._active:raise RuntimeError('This callback adapter is already active.')
        square._active,square._error = True,None
        reserved = True
    elif isinstance(function,(HessianCallback,RectangularHessianCallback)):
        raise TypeError('F requires a residual callback, not a Hessian provider.')
    try:
        step,order,kind = _check_finite_difference_args(step,order,kind)
        if order+1 > _MAX_TENSOR_RANK:
            raise ValueError('Dense derivative order exceeds the compatibility rank limit of 32.')
        point = _point_array(x)
        n = point.size
        steps = _step_array(step,n)
        scheme = 'forward' if kind == 1 else 'backward' if kind == -1 else 'central'
        if rectangular is not None:
            if rectangular._n != n:raise ValueError('Residual input dimension must match x.')
            m = int(rectangular._m)
            proxy = make_rectangular_callback(rectangular._function,n,m,
                                               rectangular._context,rectangular)
        elif square is not None:
            if square._dimension != n:raise ValueError('Residual input dimension must match x.')
            m = n
            square_bridge.function,square_bridge.context = square._function,square._context
            # The borrowed stack bridge is used only during this synchronous call.
            proxy = make_rectangular_callback(_square_residual,n,m,&square_bridge,square)
        else:
            legacy = PyNdArrayFPtr.from_f(function)
            if cached is None:base = _value_array(legacy.eval(point.copy()))
            else:base = _value_array(cached)
            m = base.size
            array_bridge = _ArrayResidual()
            array_bridge.function,array_bridge.output_dimension = legacy,m
            proxy = RectangularCallback(array_bridge,n,m)
            cached = base
        if rectangular is not None or square is not None:
            if cached is not None:cached = _value_array(cached,m)
        key = (int(n),int(m),order,kind,tuple(steps),_MAX_WORK_BYTES,
               _MAX_STENCIL_TERMS,_MAX_TENSOR_RANK) if owner is not None else None
        if owner is not None and owner._plan is not None and owner._plan_key == key:
            plan = owner._plan
        else:
            if order == 1:
                plan = RectangularJacobianPlan(n,m,scheme,steps,memory_bytes=_MAX_WORK_BYTES)
            elif order == 2:
                plan = RectangularHessianPlan(n,m,scheme,steps,memory_bytes=_MAX_WORK_BYTES)
            else:
                plan = TensorDifferencePlan(n,m,order,scheme,steps,
                    max_terms=_MAX_STENCIL_TERMS,memory_bytes=_MAX_WORK_BYTES)
        try:
            estimate = plan.evaluate(proxy,point,f_x=cached)
        except BaseException:
            # Python callbacks wrapped by the reserved original save errors there,
            # while the proxy's native status still reaches the plan correctly.
            if rectangular is not None and rectangular._error is not None:
                error = rectangular._error
                rectangular._error = None
                raise error
            if square is not None and square._error is not None:
                error = square._error
                square._error = None
                raise error
            raise
        if estimate.status == 'breakdown':
            raise ValueError('h does not produce finite, distinct sample points.')
        if estimate.status != 'ok':
            raise ValueError('the finite-difference derivative must be finite.')
        values = estimate.jacobian if order == 1 else estimate.hessian if order == 2 else estimate.derivative
        result = np.asarray(values,dtype=np.float64)
        if owner is not None:
            owner._plan,owner._plan_key = plan,key
        return result
    finally:
        if reserved:
            if rectangular is not None:rectangular._leave()
            if square is not None:square._active = False


cdef class VectorDerivativeApproximation(NdArrayFPtr):
    """Legacy ndarray derivative-provider base, also retaining native residuals."""
    def __init__(self,F):
        if isinstance(F,(RectangularCallback,VectorCallback)):
            self._native_function = F
            self.F = None
        else:
            self.F = PyNdArrayFPtr.from_f(F)
            self._native_function = None

    def __call__(self,x):
        return self.eval(_point_array(x))

    cpdef np.ndarray eval_with_f_val(self,np.ndarray x,np.ndarray F_x):
        return self.eval2(x,F_x)


cdef np.ndarray generalized_finite_difference_kernel(
        NdArrayFPtr F,
        np.ndarray[np.float64_t,ndim=1] x,
        np.ndarray[np.float64_t,ndim=1] F_x,
        np.ndarray[np.float64_t,ndim=1] h,
        int order=1,int kind=0):
    """Preserved compiled ndarray boundary; numerical execution uses native plans."""
    return _difference(F,x,F_x,h,order,kind)


cdef class GeneralizedFiniteDifference(VectorDerivativeApproximation):
    """NumPy-facing derivative wrapper retaining one validated native plan."""
    def __init__(self,F,h=1.,order=1,kind=0):
        super().__init__(F)
        h,order,kind = _check_finite_difference_args(h,order,kind)
        self.h = h if isinstance(h,float) else np.array(h,dtype=np.float64)
        self.order,self.kind = order,kind

    def __call__(self,x):
        self.n_f_calls += 1
        return _difference(self.F if self._native_function is None else self._native_function,
                           x,None,self.h,self.order,self.kind,self)

    cdef np.ndarray eval(self,np.ndarray x):
        self.n_f_calls += 1
        return _difference(self.F if self._native_function is None else self._native_function,
                           x,None,self.h,self.order,self.kind,self)

    cdef np.ndarray eval2(self,np.ndarray x,np.ndarray F_x):
        self.n_f_calls += 1
        return _difference(self.F if self._native_function is None else self._native_function,
                           x,F_x,self.h,self.order,self.kind,self)


@register('cyroot.da.vector')
@cython.binding(True)
def generalized_finite_difference(F: Callable[[VectorLike],VectorLike],
                                  x: VectorLike,F_x: Optional[ArrayLike]=None,
                                  h: Union[float,VectorLike]=1.,order=1,
                                  kind: Union[int,str]=0):
    """Estimate a dense derivative with native rounded-node finite differences.

    Return a C-contiguous float64 ndarray of shape (m,)+(n,)*order. Ordinary
    callbacks receive independent ndarray snapshots; explicit residual-kind
    RectangularCallback/VectorCallback adapters use their native paths. Cached
    F_x avoids the base evaluation. Mutable wrapper options are revalidated.

    Central steps use x+-h rather than the legacy first-derivative half steps.
    Orders1/2 use native Jacobian/Hessian plans; higher orders share the native
    tensor plan's rounded-node weights and unique sample cache. Compiled native
    callbacks execute without the GIL; Python callbacks, setup and exports use it.
    Rank32,1m stencil-term and256MiB native logical memory limits apply; Python
    planning/export and NumPy boundary copies use additional memory. See
    docs/vector-derivative-migration.md for numerical changes and failure policy.
    """
    return _difference(F,x,F_x,h,order,kind)
