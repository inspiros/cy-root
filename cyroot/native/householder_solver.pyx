# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Resumable Householder with runtime-order native reciprocal recurrence."""
from cyroot.defaults cimport ETOL, ERTOL, PTOL, PRTOL, MAX_ITER
from typing import NamedTuple
from libc.math cimport fabs, isfinite, NAN, INFINITY
from libc.string cimport memset

from .scalar cimport ScalarCallback
from .scalar import NativeCallbackError
from .scalar_derivatives cimport (
    ScalarDerivativePlan, ScalarSecondDerivativePlan, ScalarResidualFunction,
    SCALAR_DIFF_CALLBACK_FAILURE, SCALAR_DIFF_NONFINITE, SCALAR_DIFF_BREAKDOWN,
)
from .higher_derivatives cimport (
    ScalarFiniteDifferencePlan, compute_scalar_difference,
)
from .higher_derivatives import _checked_integer
from .householder cimport householder_step
from cyroot._check_args import _check_stop_cond_args
from cyroot.return_types cimport NewtonMethodsReturnType
from cyroot.return_types import _normalize_reason
from cyroot._stopping cimport (
    init_stop, advance_x, check_stop, RUNNING, MAX_ITERATIONS,
    BREAKDOWN, NONFINITE, CALLBACK_FAILURE,
)


cdef enum:
    FUNCTION = 1
    DERIVATIVE = 2


class HouseholderStep(NamedTuple):
    """Copied immutable point, derivative tuple, counters and diagnostics."""
    root: float
    f_root: float
    df_root: tuple
    iters: int
    f_calls: tuple
    precision: float
    error: float
    termination_reason: object
    callback_code: int
    callback_stage: object
    callback_order: object


cdef object _stage_name(int stage):
    if stage == FUNCTION:
        return 'function'
    if stage == DERIVATIVE:
        return 'derivative'
    return None


cdef class Householder:
    """Runtime-order Householder with analytic or native numerical derivatives.

    derivatives is an exact list/tuple of providers in increasing derivative
    order; None entries choose native finite differences. derivatives=None
    requires order and chooses all numerical derivatives. step affects only
    default numerical entries. Individual plans retain their configured step.
    """

    def __init__(self, function, derivatives, x0, f_x0=None, dfs_x0=None, *,
                 order=None, step=None, etol=ETOL,
                 ertol=ERTOL, ptol=PTOL,
                 prtol=PRTOL, max_iter=MAX_ITER,
                 history=0, history_every=1, history_bytes=8 * 1024 * 1024,
                 workspace_bytes=8 * 1024 * 1024, max_nodes=65):
        cdef ScalarCallback adapter, residual
        cdef ScalarFiniteDifferencePlan plan, normalized_step_plan
        cdef ScalarDerivativePlan first_plan
        cdef ScalarSecondDerivativePlan second_plan
        cdef size_t k, index, span_native
        cdef double initial, f_atol, f_rtol, x_atol, x_rtol
        if self._constructing or self._adapters is not None or self._initialized:
            raise RuntimeError('Solvers cannot be reinitialized.')
        self._constructing = True
        try:
            if derivatives is not None and type(derivatives) not in (tuple, list):
                raise TypeError('derivatives must be an exact tuple/list or None.')
            # Exact containers have no user-defined iteration during discovery.
            explicit = None if derivatives is None else tuple(derivatives)
            residual = function if isinstance(function, ScalarCallback) else ScalarCallback(function)
            adapters = [residual]
            providers = []
            if explicit is not None:
                for index in range(len(explicit)):
                    provider = explicit[index]
                    k = index + 1
                    # Check native types before callable adaptation, including
                    # callable subclasses whose Python properties can lie.
                    if isinstance(provider, ScalarFiniteDifferencePlan):
                        plan = provider
                        if not plan._initialized:
                            raise RuntimeError('Finite difference plans must be initialized.')
                        if plan._order != k:
                            raise ValueError('Finite difference plan has the wrong derivative order.')
                    elif isinstance(provider, ScalarDerivativePlan):
                        first_plan = provider
                        if not first_plan._initialized:
                            raise RuntimeError('Scalar derivative plans must be initialized.')
                        if k != 1:
                            raise ValueError('First derivative plan requires derivative order one.')
                    elif isinstance(provider, ScalarSecondDerivativePlan):
                        second_plan = provider
                        if not second_plan._initialized:
                            raise RuntimeError('Scalar derivative plans must be initialized.')
                        if k != 2:
                            raise ValueError('Second derivative plan requires derivative order two.')
                    elif provider is not None:
                        adapter = provider if isinstance(provider, ScalarCallback) else ScalarCallback(provider)
                        provider = adapter
                        adapters.append(adapter)
                    providers.append(provider)
            self._adapters = tuple(adapters)
            self._providers = tuple(providers)
            self._enter()
            try:
                # Every coercion, native allocation and callback below happens
                # with all discovered callback aliases reserved together.
                byte_limit = _checked_integer(workspace_bytes, 'workspace_bytes')
                node_limit = _checked_integer(max_nodes, 'max_nodes')
                if node_limit == 0:
                    raise ValueError('max_nodes must be positive.')
                if explicit is None:
                    if order is None:
                        raise ValueError('order is required when derivatives is None.')
                    degree = _checked_integer(order, 'order')
                else:
                    degree = len(explicit)
                    if order is not None and _checked_integer(order, 'order') != degree:
                        raise ValueError('order must match the number of derivative providers.')
                if degree == 0:
                    raise ValueError('order must be positive.')
                span = degree + 1
                # These are Python-integer products, checked before allocation
                # and before constructing a potentially enormous plan list.
                minimum_bytes = span * (sizeof(double) + sizeof(size_t) +
                    sizeof(HouseholderRole) + sizeof(unsigned char) +
                    2 * sizeof(ScaledCoefficient))
                if (minimum_bytes > byte_limit or span > self._values.max_size() or
                        span > self._calls.max_size() or span > self._roles.max_size() or
                        span > self._cached.max_size() or
                        2 * span > self._recurrence.max_size()):
                    raise ValueError('Requested order exceeds workspace_bytes or native limits.')
                if explicit is None:
                    providers = [None] * degree
                defaults = any(provider is None for provider in providers)
                normalized_step = None
                if defaults and step is not None:
                    normalized_step_plan = ScalarFiniteDifferencePlan(1, step=step)
                    normalized_step = normalized_step_plan._step
                native_providers = []
                largest_work = 0
                for index in range(degree):
                    provider = providers[index]
                    k = index + 1
                    if provider is None:
                        plan = ScalarFiniteDifferencePlan(k, step=normalized_step,
                            max_nodes=node_limit, workspace_bytes=byte_limit)
                        provider = plan
                    elif isinstance(provider, ScalarDerivativePlan):
                        first_plan = provider
                        plan = ScalarFiniteDifferencePlan(k,
                            ('forward', 'backward', 'central')[first_plan._scheme],
                            None if first_plan._automatic else first_plan._step,
                            workspace_bytes=byte_limit)
                        provider = plan
                    elif isinstance(provider, ScalarSecondDerivativePlan):
                        second_plan = provider
                        plan = ScalarFiniteDifferencePlan(k,
                            ('forward', 'backward', 'central')[second_plan._scheme],
                            None if second_plan._automatic else second_plan._step,
                            workspace_bytes=byte_limit)
                        provider = plan
                    if isinstance(provider, ScalarFiniteDifferencePlan):
                        plan = provider
                        largest_work = max(largest_work, plan._work)
                    native_providers.append(provider)
                required_bytes = minimum_bytes + largest_work * sizeof(double)
                if (required_bytes > byte_limit or
                        largest_work > self._difference_workspace.max_size()):
                    raise ValueError('Requested numerical scratch exceeds workspace_bytes or native limits.')
                # Keep original specialized plans as well as their POD-source
                # generic counterparts alive. No providers are accessed nogil.
                self._providers = tuple(providers) + tuple(native_providers)

                limit = _checked_integer(history, 'history')
                every = _checked_integer(history_every, 'history_every')
                history_limit = _checked_integer(history_bytes, 'history_bytes')
                if every == 0:
                    raise ValueError('history_every must be positive.')
                record_bytes = (sizeof(HouseholderRecord) + degree * sizeof(double) +
                                span * sizeof(size_t))
                if (limit * record_bytes > history_limit or
                        limit > self._records.max_size() or
                        limit * degree > self._history_derivatives.max_size() or
                        limit * span > self._history_calls.max_size()):
                    raise ValueError('Requested history exceeds history_bytes or native limits.')
                if max_iter is not None:
                    max_iter = _checked_integer(max_iter, 'max_iter')
                etol, ertol, ptol, prtol, max_iter = _check_stop_cond_args(
                    etol, ertol, ptol, prtol, max_iter)
                initial = x0
                f_atol, f_rtol, x_atol, x_rtol = etol, ertol, ptol, prtol
                self._order = degree
                self._max_iter = max_iter
                self._history_every = every
                span_native = span
                self._values.resize(span_native)
                self._calls.resize(span_native)
                self._roles.resize(span_native)
                self._cached.resize(span_native)
                self._recurrence.resize(2 * span_native)
                self._difference_workspace.resize(largest_work)
                self._records.resize(limit)
                self._history_derivatives.resize(limit * degree)
                self._history_calls.resize(limit * span)
                for k in range(span_native):
                    self._values[k] = NAN
                    self._calls[k] = 0
                    self._cached[k] = 0
                self._roles[0].function = residual._function
                self._roles[0].context = residual._context
                self._roles[0].numerical = False
                for index in range(self._order):
                    k = index + 1
                    provider = native_providers[index]
                    if isinstance(provider, ScalarFiniteDifferencePlan):
                        plan = provider
                        plan.copy_config(&self._roles[k].difference)
                        self._roles[k].numerical = True
                    else:
                        adapter = provider
                        self._roles[k].function = adapter._function
                        self._roles[k].context = adapter._context
                        self._roles[k].numerical = False
                self._cached[0] = f_x0 is not None
                if f_x0 is not None:
                    self._values[0] = f_x0
                if dfs_x0 is not None:
                    if len(dfs_x0) != degree:
                        raise ValueError('dfs_x0 must have one entry per derivative order.')
                    for index in range(self._order):
                        cached = dfs_x0[index]
                        if cached is not None:
                            self._values[index+1] = cached
                            self._cached[index+1] = 1
                memset(&self._state, 0, sizeof(HouseholderRecord))
                self._state.root = initial
                self._state.f_root = self._values[0]
                self._state.precision = INFINITY
                self._state.error = fabs(self._values[0])
                init_stop(&self._stop, 0., fabs(initial),
                          f_atol, f_rtol, x_atol, x_rtol)
                with nogil:
                    self._initialize()
                    self._record()
                self._initialized = True
                self._raise_callback_error()
            finally:
                self._release()
        finally:
            self._constructing = False

    cdef void _ready(self) except *:
        if self._busy:
            raise RuntimeError('This solver is already running.')
        if not self._initialized:
            raise RuntimeError('This solver is not initialized.')

    cdef void _enter(self) except *:
        cdef ScalarCallback adapter
        if self._busy:
            raise RuntimeError('This solver is already running.')
        if self._adapters is None or not self._adapters:
            raise RuntimeError('Solver requires initialized callback providers.')
        # Inspect every provider before marking any active; aliases are valid.
        for candidate in self._adapters:
            adapter = candidate
            if adapter._function == NULL:
                raise RuntimeError('Solver requires initialized callback providers.')
            if adapter._active:
                raise RuntimeError('This callback adapter is already active.')
        self._busy = True
        for candidate in self._adapters:
            adapter = candidate
            adapter._active = True
            adapter._error = None

    cdef void _release(self) noexcept:
        cdef ScalarCallback adapter
        for candidate in self._adapters:
            adapter = candidate
            adapter._active = False
        self._busy = False

    cdef int _evaluate(self, size_t role, double x, double* out) noexcept nogil:
        cdef int code = 0, status, sample_code = 0
        out[0] = NAN
        self._calls[role] += 1
        if self._roles[role].numerical:
            status = compute_scalar_difference(
                &self._roles[role].difference, x, self._values[0], out,
                <ScalarResidualFunction>self._roles[0].function, self._roles[0].context,
                &self._calls[0], &sample_code, self._difference_workspace.data(),
                self._difference_workspace.size())
            if status == SCALAR_DIFF_CALLBACK_FAILURE:
                code = sample_code
            elif status == SCALAR_DIFF_NONFINITE:
                self._stop.reason = NONFINITE
                return status
            elif status == SCALAR_DIFF_BREAKDOWN:
                self._stop.reason = BREAKDOWN
                return status
        else:
            code = self._roles[role].function(x, out, self._roles[role].context)
        if code != 0:
            self._state.callback_code = code
            self._state.callback_stage = FUNCTION if role == 0 else DERIVATIVE
            self._state.callback_order = role
            self._stop.reason = CALLBACK_FAILURE
        return code

    cdef void _clear_derivatives(self, size_t start) noexcept nogil:
        cdef size_t k
        for k in range(start, self._order+1):
            self._values[k] = NAN
            self._cached[k] = 0

    cdef void _derivatives(self) noexcept nogil:
        cdef size_t k
        cdef double value
        for k in range(1, self._order+1):
            value = self._values[k]
            if not self._cached[k]:
                if self._evaluate(k, self._state.root, &value) != 0:
                    self._clear_derivatives(k)
                    return
            if not isfinite(value):
                self._clear_derivatives(k)
                self._stop.reason = NONFINITE
                return
            self._values[k] = value

    cdef void _initialize(self) noexcept nogil:
        cdef double value = NAN
        if not isfinite(self._state.root):
            self._stop.reason = NONFINITE
            return
        if not self._cached[0]:
            if self._evaluate(0, self._state.root, &value) != 0:
                return
            self._state.f_root = self._values[0] = value
        self._state.error = fabs(self._values[0])
        self._stop.f_scale = self._state.error
        if check_stop(&self._stop, self._state.error, INFINITY, False):
            return
        self._derivatives()

    cdef void _check(self) noexcept nogil:
        if check_stop(&self._stop, self._state.error, self._state.precision,
                      self._state.iters > 0):
            return
        if self._max_iter != 0 and self._state.iters >= self._max_iter:
            self._stop.reason = MAX_ITERATIONS

    cdef void _record(self) noexcept nogil:
        cdef size_t record_index, k, derivative_offset, calls_offset
        self._state.reason = <int>self._stop.reason
        if self._records.size() == 0:
            return
        if self._record_count > 0 and self._records[self._record_count-1].iters == self._state.iters:
            record_index = self._record_count-1
        else:
            if self._state.iters % self._history_every != 0 and self._stop.reason == RUNNING:
                return
            if self._record_count >= self._records.size():
                self._history_truncated = True
                return
            record_index = self._record_count
            self._record_count += 1
        self._records[record_index] = self._state
        derivative_offset = record_index * self._order
        calls_offset = record_index * (self._order+1)
        for k in range(self._order):
            self._history_derivatives[derivative_offset+k] = self._values[k+1]
        for k in range(self._order+1):
            self._history_calls[calls_offset+k] = self._calls[k]

    cdef void _step(self) noexcept nogil:
        cdef double delta = NAN, candidate, precision, value = NAN
        cdef int status
        if self._stop.reason != RUNNING:
            return
        status = householder_step(self._values.data(), self._order,
            self._recurrence.data(), self._recurrence.size(), &delta)
        if status != 0:
            self._stop.reason = NONFINITE if status == 1 else BREAKDOWN
            self._record()
            return
        candidate = self._state.root + delta
        precision = fabs(candidate-self._state.root)
        if not isfinite(delta) or not isfinite(candidate) or not isfinite(precision):
            self._stop.reason = NONFINITE
            self._record()
            return
        self._state.iters += 1
        if self._evaluate(0, candidate, &value) == 0:
            if not isfinite(value):
                self._stop.reason = NONFINITE
            else:
                self._state.root = candidate
                self._state.f_root = self._values[0] = value
                self._clear_derivatives(1)
                self._state.error = fabs(value)
                self._state.precision = precision
                advance_x(&self._stop, fabs(candidate))
                self._check()
                if self._stop.reason == RUNNING:
                    self._derivatives()
        self._record()

    cdef void _raise_callback_error(self) except *:
        cdef ScalarCallback adapter
        for candidate in self._adapters:
            adapter = candidate
            if adapter._error is not None:
                error = adapter._error
                adapter._error = None
                raise error
        if self._stop.reason == CALLBACK_FAILURE:
            error = NativeCallbackError(self._state.callback_code, self._copy_result(),
                                       _stage_name(self._state.callback_stage))
            error.callback_order = self._state.callback_order
            raise error

    cdef object _snapshot(self, HouseholderRecord record,
                          const double* derivatives, const size_t* calls):
        return HouseholderStep(
            record.root, record.f_root,
            tuple([derivatives[k] for k in range(self._order)]), record.iters,
            tuple([calls[k] for k in range(self._order+1)]),
            record.precision, record.error,
            None if record.reason == RUNNING else _normalize_reason(record.reason),
            record.callback_code, _stage_name(record.callback_stage),
            None if record.callback_stage == 0 else record.callback_order)

    cdef object _copy_result(self):
        cdef HouseholderRecord state = self._state
        return NewtonMethodsReturnType(
            state.root, state.f_root,
            tuple([self._values[k+1] for k in range(self._order)]), state.iters,
            tuple([self._calls[k] for k in range(self._order+1)]),
            state.precision, state.error, False, False,
            termination_reason=None if state.reason == RUNNING else state.reason)

    def step(self):
        """Attempt one finite candidate residual and copy the resulting state."""
        self._ready()
        self._enter()
        try:
            if self._stop.reason == RUNNING:
                with nogil:
                    self._step()
                self._raise_callback_error()
            return self._snapshot(self._state, self._values.data()+1, self._calls.data())
        finally:
            self._release()

    def run(self):
        """Use the remaining total budget through the native step loop."""
        self._ready()
        self._enter()
        try:
            if self._stop.reason == RUNNING:
                with nogil:
                    while self._stop.reason == RUNNING:
                        self._step()
                self._raise_callback_error()
            return self._copy_result()
        finally:
            self._release()

    @property
    def result(self):
        self._ready()
        self._enter()
        try:
            return self._copy_result()
        finally:
            self._release()

    @property
    def history(self):
        cdef size_t i
        self._ready()
        self._enter()
        try:
            return tuple([self._snapshot(self._records[i],
                self._history_derivatives.data()+i*self._order,
                self._history_calls.data()+i*(self._order+1))
                for i in range(self._record_count)])
        finally:
            self._release()

    @property
    def done(self):
        self._ready()
        return self._stop.reason != RUNNING

    @property
    def order(self):
        self._ready()
        return self._order

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
        return _stage_name(self._state.callback_stage)

    @property
    def callback_order(self):
        self._ready()
        return None if self._state.callback_stage == 0 else self._state.callback_order

    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration
        return self.step()


def householder(function, derivatives, x0, **options):
    """Run Householder; retain a Householder instance for resumption/history."""
    return Householder(function, derivatives, x0, **options).run()
