# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Generalized bisection without Python operations in the native solver loop."""
from typing import NamedTuple
from libc.limits cimport INT_MAX
from libc.math cimport fabs, fmax, isfinite, INFINITY, NAN
from libc.string cimport memcpy, memset
from ._buffers cimport DoubleBuffer, is_native_double_buffer
from .buffers cimport OwnedDoubleBuffer, MatrixView, VectorView
from .vector cimport VectorCallback
from .rectangular cimport RectangularCallback
from .polygons cimport construct_polygon, polygon_workspace_size
from .scalar import NativeCallbackError
from .higher_derivatives import _checked_integer
from cyroot._check_args import _check_stop_cond_args
from cyroot.return_types import BracketingMethodsReturnType, _normalize_reason
from cyroot._stopping cimport (
    init_stop, check_stop, residual_met, RUNNING, RESIDUAL_TOLERANCE,
    MAX_ITERATIONS, NONFINITE, CALLBACK_FAILURE, BREAKDOWN, STAGNATION,
)


class VrahatisStep(NamedTuple):
    """Immutable copied estimate, polygon and diagnostics after a sweep."""
    root: tuple
    f_root: tuple
    bracket: tuple
    f_bracket: tuple
    iters: int
    f_calls: int
    precision: float
    error: float
    termination_reason: object
    callback_code: int

    @property
    def success(self):
        return self.termination_reason in ('residual_tolerance', 'bracket_tolerance')

    @property
    def converged(self):return self.success

    @property
    def optimal(self):return self.termination_reason == 'residual_tolerance'


cdef void _fill(double* data, size_t count, double value) noexcept nogil:
    cdef size_t i
    for i in range(count):data[i] = value


cdef double _norm(const double* data, size_t count) noexcept nogil:
    cdef size_t i
    cdef double value = 0.
    for i in range(count):
        if not isfinite(data[i]):return INFINITY
        value = fmax(value, fabs(data[i]))
    return value


cdef object _vector_tuple(const double* data, size_t count):
    return tuple([data[i] for i in range(count)])


cdef object _rows_tuple(const double* data, size_t rows, size_t columns):
    return tuple([_vector_tuple(data+i*columns, columns) for i in range(rows)])


cdef object _copy_rows(object source, object workspace_limit):
    """Copy a vector or row matrix while the original adapter is reserved."""
    cdef DoubleBuffer guard
    cdef VectorView v
    cdef MatrixView a
    cdef size_t i, j
    if isinstance(source, (str, bytes, bytearray)):
        raise TypeError('Expected a numeric point or matrix of points.')
    if is_native_double_buffer(source):
        guard = DoubleBuffer(source)
        if guard._ndim == 1:
            v = guard.vector_view()
            if int(v.size())*sizeof(double) > workspace_limit:
                raise ValueError('Input storage exceeds workspace_bytes.')
            return (_vector_tuple_view(v),)
        if guard._ndim != 2:raise ValueError('Expected a vector or row matrix.')
        a = guard.matrix_view()
        if int(a.rows())*int(a.columns())*sizeof(double) > workspace_limit:
            raise ValueError('Input storage exceeds workspace_bytes.')
        return tuple([tuple([a.get(i,j) for j in range(a.columns())]) for i in range(a.rows())])
    count = len(source)
    if count == 0:raise ValueError('Initial points/residuals must be nonempty.')
    if isinstance(source[0], (str, bytes, bytearray)):
        raise TypeError('Expected numeric coordinates.')
    try:width = len(source[0])
    except TypeError:
        if count*sizeof(double) > workspace_limit:raise ValueError('Input storage exceeds workspace_bytes.')
        return (tuple([float(source[i]) for i in range(count)]),)
    if count*width*sizeof(double) > workspace_limit:raise ValueError('Input storage exceeds workspace_bytes.')
    result = []
    for row in source:
        if isinstance(row, (str, bytes, bytearray)) or len(row) != width:
            raise ValueError('All point/residual rows must have the same length.')
        result.append(tuple([float(row[j]) for j in range(width)]))
    return tuple(result)


cdef object _vector_tuple_view(VectorView view):
    return tuple([view.get(i) for i in range(view.size())])


cdef int _polygon_residual(const double* x, size_t n, size_t m,
                          double* out, void* context) noexcept nogil:
    # Polygon construction owns its counter; the shared evaluator increments
    # the solver counter and preserves original callback errors/status.
    return (<Vrahatis><object>context)._polygon_evaluate(x,out)


cdef class Vrahatis:
    """Retain a generalized-bisection state and inspect copied sweep history.

    A complete set of vertex signs is a combinatorial starting condition.
    For multiple residual components it does not certify a root, so a narrow
    polygon reports stagnation unless the residual tolerance is satisfied.
    """
    def __init__(self, function, x0s, F_x0s=None, *, h=None, etol=1e-10,
                 ertol=8.881784197001252e-16, ptol=1e-12,
                 prtol=8.881784197001252e-16, max_iter=200, history=0,
                 history_every=1, history_bytes=8*1024*1024,
                 workspace_bytes=64*1024*1024, polygon_eps=1e-5,
                 max_evaluations=100000, max_visits=1000000, max_bisections=256):
        cdef size_t i, j
        cdef double* data
        cdef bint discovered = False
        if (self._constructing or self._initialized or self._storage != NULL or
                self._function is not None or self._rect_function is not None):
            raise RuntimeError('Solvers cannot be reinitialized.')
        self._constructing = True
        try:
            if isinstance(function, VectorCallback):
                self._function = function
                if self._function._function == NULL or self._function._constructing:
                    raise RuntimeError('Residual callback is not initialized.')
                if self._function._jacobian:raise ValueError('function requires a residual-kind callback.')
            elif isinstance(function, RectangularCallback):
                self._rect_function = function
                if self._rect_function._function == NULL or self._rect_function._constructing:
                    raise RuntimeError('Residual callback is not initialized.')
                if self._rect_function._jacobian:raise ValueError('function requires a residual-kind callback.')
            elif not callable(function):raise TypeError('function must be callable.')
            self._enter()
            try:
                etol, ertol, ptol, prtol, max_iter = _check_stop_cond_args(etol, ertol, ptol, prtol, max_iter)
                limit = _checked_integer(history, 'history')
                every = _checked_integer(history_every, 'history_every')
                byte_limit = _checked_integer(history_bytes, 'history_bytes')
                workspace_limit = _checked_integer(workspace_bytes, 'workspace_bytes')
                evaluation_limit = _checked_integer(max_evaluations, 'max_evaluations')
                visit_limit = _checked_integer(max_visits, 'max_visits')
                bisection_limit = _checked_integer(max_bisections, 'max_bisections')
                if every == 0:raise ValueError('history_every must be positive.')
                if evaluation_limit == 0 or visit_limit == 0 or bisection_limit == 0:
                    raise ValueError('Polygon search budgets must be positive.')
                polygon_eps = float(polygon_eps)
                if not isfinite(polygon_eps) or polygon_eps <= 0:
                    raise ValueError('polygon_eps must be finite and positive.')
                rows = _copy_rows(x0s, workspace_limit)
                dimension = len(rows[0])
                pool = len(rows)
                if dimension == 0 or dimension > INT_MAX:
                    raise ValueError('Points must be nonempty with dimension fitting C int.')
                self._single = pool == 1
                output_dimension = None
                if self._function is not None:
                    if self._function._dimension != dimension:raise ValueError('Callback input dimension must match points.')
                    output_dimension = int(self._function._dimension)
                elif self._rect_function is not None:
                    if self._rect_function._n != dimension:raise ValueError('Callback input dimension must match points.')
                    output_dimension = int(self._rect_function._m)
                cached = None
                if F_x0s is not None:
                    cached = _copy_rows(F_x0s, workspace_limit)
                    if len(cached) != pool:raise ValueError('Cached residuals must have one row per point.')
                    cached_dimension = len(cached[0])
                    if output_dimension is not None and output_dimension != cached_dimension:
                        raise ValueError('Cached residual dimension must match the callback.')
                    output_dimension = cached_dimension
                base_value = None
                if output_dimension is None:
                    finite = all(isfinite(value) for row in rows for value in row)
                    if finite:
                        # There is no original adapter for an ordinary callable.
                        # Dimensional discovery and conversion occur under the
                        # solver's construction reservation, before publication.
                        base_value = function(rows[0])
                        discovered = True
                        copied_value = _copy_rows(base_value, workspace_limit)
                        if len(copied_value) != 1:raise ValueError('F must return a vector.')
                        base_value = copied_value[0]
                        output_dimension = len(base_value)
                    else:output_dimension = dimension
                if output_dimension == 0 or output_dimension >= sizeof(size_t)*8:
                    raise ValueError('Residual dimension exceeds native polygon sign capacity.')
                expected = 1 << output_dimension
                capacity = max(pool, expected)
                polygon_count = polygon_workspace_size(dimension, output_dimension) if self._single else 0
                stride = (capacity+1)*(dimension+output_dimension)
                workspace_count = stride+capacity*(dimension+output_dimension)+dimension+output_dimension+dimension+polygon_count
                record_bytes = sizeof(VrahatisRecord)+stride*sizeof(double)
                if (workspace_count > OwnedDoubleBuffer.max_size() or capacity > self._seen.max_size() or
                        workspace_count*sizeof(double)+capacity*sizeof(unsigned char) > workspace_limit):
                    raise ValueError('Requested polygon workspace exceeds workspace_bytes or native limits.')
                if (limit > self._records.max_size() or limit*stride > OwnedDoubleBuffer.max_size() or
                        limit > byte_limit//record_bytes):
                    raise ValueError('Requested history exceeds history_bytes or native limits.')
                self._n, self._m, self._k = dimension, output_dimension, expected
                self._capacity, self._count, self._source_count = capacity, pool, pool
                self._stride, self._max_iter, self._history_every = stride, max_iter, every
                self._polygon_eps = polygon_eps
                self._max_evaluations, self._max_visits = evaluation_limit, visit_limit
                self._max_bisections = bisection_limit
                if self._function is None and self._rect_function is None:
                    self._rect_function = RectangularCallback(function, self._n, self._m)
                    self._rect_function._active = True
                self._storage = new OwnedDoubleBuffer(workspace_count)
                data = self._storage.data()
                self._x, self._f = data, data+self._n
                self._points = self._f+self._m
                self._values = self._points+self._capacity*self._n
                self._source_points = data+self._stride
                self._source_values = self._source_points+self._capacity*self._n
                self._trial = self._source_values+self._capacity*self._m
                self._trial_f = self._trial+self._n
                self._h = self._trial_f+self._m
                self._polygon_work = self._h+self._n
                _fill(data, workspace_count, NAN)
                self._history_data = new OwnedDoubleBuffer(limit*stride)
                self._records.resize(limit)
                self._seen.resize(capacity)
                for i in range(self._source_count):
                    for j in range(self._n):self._source_points[i*self._n+j] = rows[i][j]
                if cached is not None:
                    for i in range(self._source_count):
                        for j in range(self._m):self._source_values[i*self._m+j] = cached[i][j]
                elif base_value is not None:
                    for j in range(self._m):self._source_values[j] = base_value[j]
                if self._single:
                    if h is None:
                        # Infinity-norm scaling stays finite even where a
                        # rounded sum of divided magnitudes would overflow.
                        mean = _norm(self._source_points,self._n)
                        # Invalid initial coordinates are classified by native
                        # initialization before sampling, not step validation.
                        mean = fmax(1.,mean) if isfinite(mean) else 1.
                        for j in range(self._n):self._h[j] = mean
                    else:
                        try:len(h)
                        except TypeError:
                            value = float(h)
                            for j in range(self._n):self._h[j] = value
                        else:
                            step_rows = _copy_rows(h, workspace_limit)
                            if len(step_rows) != 1 or len(step_rows[0]) != self._n:
                                raise ValueError('h must be a scalar or vector matching the input dimension.')
                            for j in range(self._n):self._h[j] = step_rows[0][j]
                    if not isfinite(_norm(self._h, self._n)):
                        raise ValueError('h must contain finite search steps.')
                memcpy(self._x, self._source_points, self._n*sizeof(double))
                memcpy(self._points, self._source_points, self._source_count*self._n*sizeof(double))
                memcpy(self._values, self._source_values, self._source_count*self._m*sizeof(double))
                memset(&self._state, 0, sizeof(VrahatisRecord))
                self._state.f_calls = 1 if discovered else 0
                self._state.precision, self._state.error = INFINITY, INFINITY
                init_stop(&self._stop, 0., _norm(self._source_points,self._source_count*self._n), etol, ertol, ptol, prtol)
                with nogil:
                    self._initialize(cached is not None, discovered)
                    self._record()
                if self._invalid_polygon:
                    raise ValueError('Initial bounds do not form an admissible n-polygon; use every strict output sign combination exactly once.')
                self._initialized = True
                self._raise_callback_error()
            finally:self._leave()
        finally:self._constructing = False

    def __dealloc__(self):
        if self._storage != NULL:del self._storage
        if self._history_data != NULL:del self._history_data

    cdef void _ready(self) except *:
        if self._busy or self._constructing:raise RuntimeError('This solver is already active.')
        if not self._initialized:raise RuntimeError('This solver is not initialized.')

    cdef void _enter(self) except *:
        if self._busy:raise RuntimeError('This solver is already active.')
        if ((self._function is not None and self._function._active) or
                (self._rect_function is not None and self._rect_function._active)):
            raise RuntimeError('This callback adapter is already active.')
        self._busy = True
        if self._function is not None:
            self._function._active, self._function._error = True, None
        if self._rect_function is not None:
            self._rect_function._active, self._rect_function._error = True, None

    cdef void _leave(self) noexcept:
        if self._function is not None:self._function._active = False
        if self._rect_function is not None:self._rect_function._active = False
        self._busy = False

    cdef int _evaluate(self, const double* point, double* out) noexcept nogil:
        cdef int code
        _fill(out, self._m, NAN)
        self._state.f_calls += 1
        if self._rect_function is not None:code = self._rect_function.evaluate_raw(point,out)
        else:code = self._function._function(point,self._n,out,self._function._context)
        if code != 0:
            _fill(out,self._m,NAN)
            self._state.callback_code = code
            self._stop.reason = CALLBACK_FAILURE
        return code

    cdef int _polygon_evaluate(self,const double* x,double* out) noexcept nogil:
        cdef int code = self._evaluate(x,out)
        if code != 0:return code
        # Private early exit from polygon sampling after a finite converged
        # sample. The raw helper's callback-stop status is consumed below;
        # this sentinel never becomes a user callback failure.
        if residual_met(&self._stop,_norm(out,self._m)):
            memcpy(self._x,x,self._n*sizeof(double))
            memcpy(self._f,out,self._m*sizeof(double))
            self._state.error = _norm(out,self._m)
            self._polygon_root = True
            return 1
        return 0

    cdef double _geometry(self) noexcept nogil:
        cdef size_t i,j
        cdef double lower,upper,width=0., difference
        for j in range(self._n):
            lower = upper = self._points[j]
            for i in range(1,self._count):
                lower = min(lower,self._points[i*self._n+j])
                upper = max(upper,self._points[i*self._n+j])
            difference = upper-lower
            if not isfinite(difference):return INFINITY
            width = fmax(width,difference)
        return width

    cdef void _best(self) noexcept nogil:
        cdef size_t i,best=0
        cdef double error,best_error=INFINITY
        for i in range(self._count):
            error = _norm(self._values+i*self._m,self._m)
            if error < best_error:
                best = i
                best_error = error
        if isfinite(best_error):
            memcpy(self._x,self._points+best*self._n,self._n*sizeof(double))
            memcpy(self._f,self._values+best*self._m,self._m*sizeof(double))
            self._state.error = best_error

    cdef void _canonicalize(self) noexcept nogil:
        cdef size_t i,j,slot
        for i in range(self._capacity):self._seen[i] = 0
        if self._source_count != self._k:
            self._invalid_polygon = True
            return
        for i in range(self._source_count):
            slot = 0
            for j in range(self._m):
                if self._source_values[i*self._m+j] == 0.:
                    self._invalid_polygon = True
                    return
                slot = (slot<<1) | <size_t>(self._source_values[i*self._m+j]>0.)
            if self._seen[slot]:
                self._invalid_polygon = True
                return
            self._seen[slot] = 1
        # The complete input is valid before any canonical row is committed.
        for i in range(self._source_count):
            slot = 0
            for j in range(self._m):slot = (slot<<1)|<size_t>(self._source_values[i*self._m+j]>0.)
            memcpy(self._points+slot*self._n,self._source_points+i*self._n,self._n*sizeof(double))
            memcpy(self._values+slot*self._m,self._source_values+i*self._m,self._m*sizeof(double))
        self._count = self._k

    cdef void _initialize(self, bint cached, bint discovered) noexcept nogil:
        cdef size_t i,evaluations=0,visits=0,found=0
        cdef int status,code=0
        cdef double error
        if not isfinite(_norm(self._source_points,self._source_count*self._n)):
            self._stop.reason = NONFINITE
            return
        self._state.precision = self._geometry()
        for i in range(self._source_count):
            if not cached and not(discovered and i==0):
                if self._evaluate(self._source_points+i*self._n,self._trial_f) != 0:return
                if not isfinite(_norm(self._trial_f,self._m)):
                    self._stop.reason = NONFINITE
                    return
                memcpy(self._source_values+i*self._m,self._trial_f,self._m*sizeof(double))
                memcpy(self._values+i*self._m,self._trial_f,self._m*sizeof(double))
            error = _norm(self._source_values+i*self._m,self._m)
            if not isfinite(error):
                self._stop.reason = NONFINITE
                return
            self._stop.f_scale = fmax(self._stop.f_scale,error)
            if error < self._state.error:
                memcpy(self._x,self._source_points+i*self._n,self._n*sizeof(double))
                memcpy(self._f,self._source_values+i*self._m,self._m*sizeof(double))
                self._state.error = error
        # A single point's width is not a bracket or a stagnation event.
        if self._single:
            if residual_met(&self._stop,self._state.error):
                self._stop.reason = RESIDUAL_TOLERANCE
                return
            status = construct_polygon(_polygon_residual,<void*>self,
                self._source_points,self._source_values,self._h,self._n,self._m,self._polygon_eps,
                self._max_evaluations,self._max_visits,self._max_bisections,self._points,self._values,
                self._polygon_work,&evaluations,&visits,&found,&code)
            if self._polygon_root:
                self._count = 1
                memcpy(self._points,self._x,self._n*sizeof(double))
                memcpy(self._values,self._f,self._m*sizeof(double))
                self._state.precision = 0.
                self._stop.reason = RESIDUAL_TOLERANCE
                return
            if status != 0:
                # Construction scratch must not replace the last finite
                # initial state if the search fails or a callback raises.
                self._count = 1
                memcpy(self._points,self._source_points,self._n*sizeof(double))
                memcpy(self._values,self._source_values,self._m*sizeof(double))
                self._stop.reason = CALLBACK_FAILURE if status == 3 else NONFINITE if status == 2 else BREAKDOWN
                if status == 3:self._state.callback_code = code
                return
            self._count = self._k
            self._stop.f_scale = _norm(self._values,self._k*self._m)
            self._best()
        else:
            if residual_met(&self._stop,self._state.error):
                self._stop.reason = RESIDUAL_TOLERANCE
                return
            self._canonicalize()
            if self._invalid_polygon:return
            self._best()
        self._state.precision = self._geometry()
        self._stop.x_scale = _norm(self._points,self._count*self._n)
        self._check()

    cdef void _check(self) noexcept nogil:
        if check_stop(&self._stop,self._state.error,self._state.precision,
                      self._state.iters>0,self._m==1):return
        if self._max_iter != 0 and self._state.iters >= self._max_iter:
            self._stop.reason = MAX_ITERATIONS

    cdef void _step(self) noexcept nogil:
        cdef size_t p,q,bit,j,replacement
        cdef bint updated=False,same
        cdef double error
        if self._stop.reason != RUNNING:return
        self._state.iters += 1
        for p in range(self._k):
            bit = 1
            while bit < self._k:
                q = p^bit
                bit <<= 1
                if q < p:continue
                for j in range(self._n):
                    self._trial[j] = .5*self._points[p*self._n+j]+.5*self._points[q*self._n+j]
                if not isfinite(_norm(self._trial,self._n)):
                    self._stop.reason = NONFINITE
                    self._record()
                    return
                if self._evaluate(self._trial,self._trial_f) != 0:
                    self._record()
                    return
                error = _norm(self._trial_f,self._m)
                if not isfinite(error):
                    self._stop.reason = NONFINITE
                    self._record()
                    return
                if residual_met(&self._stop,error):
                    memcpy(self._x,self._trial,self._n*sizeof(double))
                    memcpy(self._f,self._trial_f,self._m*sizeof(double))
                    self._state.error = error
                    self._stop.reason = RESIDUAL_TOLERANCE
                    self._record()
                    return
                # Clear zero component bits. This is the first canonical slot
                # compatible with all strict signs, without a sign-row scan.
                replacement = 0
                for j in range(self._m):replacement=(replacement<<1)|<size_t>(self._trial_f[j]>0.)
                same = True
                for j in range(self._n):
                    if self._trial[j] != self._points[replacement*self._n+j]:same=False
                if same:
                    self._stop.reason = STAGNATION
                    self._record()
                    return
                memcpy(self._points+replacement*self._n,self._trial,self._n*sizeof(double))
                memcpy(self._values+replacement*self._m,self._trial_f,self._m*sizeof(double))
                updated = True
                self._best()
                self._state.precision = self._geometry()
                self._stop.x_scale = _norm(self._points,self._count*self._n)
                # max_iter counts complete sweeps, not individual edges.
                if check_stop(&self._stop,self._state.error,self._state.precision,True,self._m==1):
                    self._record()
                    return
        if not updated:self._stop.reason = BREAKDOWN
        self._check()
        self._record()

    cdef void _record(self) noexcept nogil:
        cdef size_t slot
        self._state.reason = <int>self._stop.reason
        self._state.count = self._count
        if self._records.size() == 0:return
        if self._record_count>0 and self._records[self._record_count-1].iters == self._state.iters:
            slot = self._record_count-1
        else:
            if self._state.iters%self._history_every != 0 and self._stop.reason == RUNNING:return
            if self._record_count >= self._records.size():
                self._history_truncated = True
                return
            slot = self._record_count
            self._record_count += 1
        self._records[slot] = self._state
        memcpy(self._history_data.data()+slot*self._stride,self._x,self._stride*sizeof(double))

    cdef void _raise_callback_error(self) except *:
        if self._function is not None and self._function._error is not None:
            error = self._function._error
            self._function._error = None
            raise error
        if self._rect_function is not None and self._rect_function._error is not None:
            error = self._rect_function._error
            self._rect_function._error = None
            raise error
        if self._stop.reason == CALLBACK_FAILURE:
            raise NativeCallbackError(self._state.callback_code,self._copy_result(),'function')

    cdef object _snapshot(self,VrahatisRecord record,const double* data):
        return VrahatisStep(_vector_tuple(data,self._n),_vector_tuple(data+self._n,self._m),
            _rows_tuple(data+self._n+self._m,record.count,self._n),
            _rows_tuple(data+self._n+self._m+self._capacity*self._n,record.count,self._m),
            record.iters,record.f_calls,record.precision,record.error,
            None if record.reason==RUNNING else _normalize_reason(record.reason),record.callback_code)

    cdef object _copy_result(self):
        return BracketingMethodsReturnType(_vector_tuple(self._x,self._n),_vector_tuple(self._f,self._m),
            self._state.iters,self._state.f_calls,_rows_tuple(self._points,self._count,self._n),
            _rows_tuple(self._values,self._count,self._m),self._state.precision,self._state.error,
            False,False,termination_reason=None if self._state.reason==RUNNING else self._state.reason)

    def step(self):
        self._ready()
        self._enter()
        try:
            if self._stop.reason==RUNNING:
                with nogil:self._step()
                self._raise_callback_error()
            return self._snapshot(self._state,self._x)
        finally:self._leave()

    def run(self):
        self._ready()
        self._enter()
        try:
            if self._stop.reason==RUNNING:
                with nogil:
                    while self._stop.reason==RUNNING:self._step()
                self._raise_callback_error()
            return self._copy_result()
        finally:self._leave()

    @property
    def result(self):
        self._ready()
        self._enter()
        try:return self._copy_result()
        finally:self._leave()

    @property
    def history(self):
        cdef size_t i
        self._ready()
        self._enter()
        try:return tuple([self._snapshot(self._records[i],self._history_data.data()+i*self._stride) for i in range(self._record_count)])
        finally:self._leave()

    @property
    def done(self):
        self._ready()
        return self._stop.reason!=RUNNING

    @property
    def history_truncated(self):
        self._ready()
        return bool(self._history_truncated)

    @property
    def callback_code(self):
        self._ready()
        return self._state.callback_code

    @property
    def callback_stage(self):
        self._ready()
        return 'function' if self._state.callback_code else None

    @property
    def input_dimension(self):
        self._ready()
        return self._n

    @property
    def output_dimension(self):
        self._ready()
        return self._m

    def __iter__(self):return self

    def __next__(self):
        if self.done:raise StopIteration
        return self.step()


def vrahatis(function,x0s,**options):
    """Run generalized bisection; retain Vrahatis for stepping/history."""
    return Vrahatis(function,x0s,**options).run()


def compute_admissible_n_polygon(function,x,h=None,eps=1e-5,**options):
    """Return copied vertices/signs from bounded native polygon construction.

    This searches only the requested cube and its input-coordinate edges;
    failure to discover every strict sign pattern raises ValueError.
    """
    cdef Vrahatis solver = Vrahatis(function,x,h=h,polygon_eps=eps,**options)
    cdef size_t i,j
    cdef double value
    solver._ready()
    solver._enter()
    try:
        if solver._count != solver._k or solver._stop.reason in (NONFINITE,BREAKDOWN):
            raise ValueError('Unable to find an admissible n-polygon; modify x or h, or increase search budgets.')
        # A solver may legitimately stop at a sampled root before canonicalizing
        # its initial rows. The standalone helper promises strict canonical signs.
        for i in range(solver._k):
            for j in range(solver._m):
                value = solver._values[i*solver._m+j]
                if (not isfinite(value) or value == 0. or
                        (value > 0.) != <bint>((i>>(solver._m-1-j))&1)):
                    raise ValueError('Unable to return a strict sign-complete polygon from these initial points.')
        return (_rows_tuple(solver._points,solver._count,solver._n),
                tuple([tuple([1. if (i>>(solver._m-1-j))&1 else -1. for j in range(solver._m)])
                       for i in range(solver._k)]))
    finally:solver._leave()
