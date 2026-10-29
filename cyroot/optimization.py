"""Residual objectives, native optimizers and scalar/vector root conveniences.

An optimizer's success is distinct from satisfying the original root tolerance.
Python optimizer orchestration uses the GIL; native objective evaluation can run
without it when called by a compiled consumer with native residual providers.
"""
from math import isfinite
from typing import NamedTuple

from ._check_args import ETOL, ERTOL
from .native.rectangular import RectangularCallback
from .native.scalar import NativeCallbackError
from .status import TerminationReason

from .native.levenberg_marquardt import (LevenbergMarquardt, LevenbergMarquardtResult,
                                          levenberg_marquardt)
from .native.bfgs import BFGS, BFGSResult, bfgs, LBFGS, lbfgs
from .native.levenberg_marquardt import PowellDogleg, PowellDoglegResult, powell_dogleg
from .native.objective import LeastSquaresObjective
from .native.objective_derivatives import ObjectiveDerivativePlan

__all__ = [
    'PowellDogleg',
    'PowellDoglegResult',
    'powell_dogleg',
    'LBFGS',
    'lbfgs',
    'BFGS',
    'BFGSResult',
    'bfgs',
    'LevenbergMarquardt',
    'LevenbergMarquardtResult',
    'levenberg_marquardt',
    'LeastSquaresObjective',
    'ObjectiveDerivativePlan',
    'OptimizationRootResult',
    'root_via_optimizer',
    'OptimizationResult',
    'OptimizationSolver',
    'find_optimization_root',
    'OPTIMIZATION_ROOT_FINDING_METHODS',
    'scalar_levenberg_marquardt',
    'vector_levenberg_marquardt',
    'scalar_bfgs',
    'vector_bfgs',
    'scalar_lbfgs',
    'vector_lbfgs',
    'scalar_powell_dogleg',
    'vector_powell_dogleg',
]


class OptimizationRootResult(NamedTuple):
    root: tuple
    f_root: tuple
    error: float
    success: bool
    termination_reason: str
    optimizer_success: object
    optimizer_status: object
    optimizer_result: object


def root_via_optimizer(objective, optimizer, x0, *, etol=ETOL, ertol=ERTOL,
                       optimizer_options=None):
    """Call optimizer(objective, copied_x0, **options), then check its root.

    ``objective`` is a LeastSquaresObjective. The optimizer must return an object
    with ``x``; optional ``success`` and ``status`` fields are preserved alongside
    its entire result. Pass optimizer-specific derivative/method options explicitly,
    e.g. {'method':'BFGS', 'jac':objective.gradient} for scipy.optimize.minimize.
    Relative tolerance uses the fixed initial residual infinity norm. Initial
    residual convergence returns immediately without invoking the optimizer.
    """
    if not isinstance(objective, LeastSquaresObjective):
        raise TypeError('objective must be a LeastSquaresObjective.')
    if not callable(optimizer):
        raise TypeError('optimizer must be callable.')
    etol, ertol = float(etol), float(ertol)
    if not isfinite(etol) or etol < 0 or not isfinite(ertol) or not 0 <= ertol < 1:
        raise ValueError('Root tolerances must be finite/nonnegative and ertol smaller than one.')
    # One guarded conversion/evaluation pins the exact copied point to its
    # residual. User __float__ methods must not be invoked a second time.
    start, initial = objective._initial_state(x0)
    scale = max(map(abs, initial))
    if scale <= etol or scale-etol <= ertol*scale:
        return OptimizationRootResult(start, initial, scale, True,
                                      'residual_tolerance', None, None, None)
    options = {} if optimizer_options is None else dict(optimizer_options)
    raw = optimizer(objective, start, **options)
    root, residual = objective._initial_state(raw.x)
    error = max(map(abs, residual))
    success = error <= etol or error-etol <= ertol*scale
    return OptimizationRootResult(root, residual, error, success,
        'residual_tolerance' if success else 'optimizer_stopped_without_root',
        getattr(raw, 'success', None), getattr(raw, 'status', None), raw)


# Public conveniences keep scalar/vector adaptation outside the native loops.


class OptimizationResult(NamedTuple):
    """Common copied state; diagnostics preserves the complete native snapshot."""
    root: float | tuple[float, ...]
    f_root: float | tuple[float, ...]
    iters: int
    f_calls: tuple[int, int]
    precision: float
    error: float
    success: bool
    termination_reason: TerminationReason | None
    method: str
    diagnostics: object

    @property
    def converged(self):
        return self.success

    @property
    def optimal(self):
        return self.success


_OPTIMIZATION_STATES = {
    'levenberg_marquardt': LevenbergMarquardt,
    'bfgs': BFGS,
    'lbfgs': LBFGS,
    'powell_dogleg': PowellDogleg,
}


class _PointDimension:
    """Delay user len until the objective has reserved original callbacks."""
    def __init__(self, point):
        self.point = point

    def __index__(self):
        return len(self.point)


class _OutputDimension:
    def __init__(self, callback):
        self.callback = callback

    def __index__(self):
        return self.callback.output_dimension


class OptimizationSolver:
    """Resumable public optimizer with scalar/vector callback conveniences.

    Scalar mode accepts f(float)->float and jacobian(float)->float. Vector mode
    accepts F(point)->vector and J(point)->matrix; input length is inferred,
    rectangular output length is explicit or taken from a native adapter.
    An existing LeastSquaresObjective may be reused without extra sampling.
    Native state classes remain available for compiled consumers.
    """
    __slots__ = ('_solver', '_objective', '_method', '_scalar', '_constructing', '_busy')

    def __init__(self, method: str, function, x0, *, jacobian=None, scalar=False,
                 input_dimension=None, output_dimension=None, **options):
        if getattr(self, '_constructing', False) or getattr(self, '_solver', None) is not None:
            raise RuntimeError('Solver facades cannot be reinitialized.')
        self._constructing, self._busy = True, False
        self._solver, self._objective = None, None
        try:
            if type(method) is not str or method not in _OPTIMIZATION_STATES:
                raise ValueError('Unknown optimization method; expected '+', '.join(_OPTIMIZATION_STATES)+'.')
            if type(scalar) is not bool:
                raise TypeError('scalar must be bool.')
            self._method, self._scalar = method, scalar
            if isinstance(function, LeastSquaresObjective):
                if jacobian is not None or input_dimension is not None or output_dimension is not None:
                    raise ValueError('An existing objective already defines providers and dimensions.')
                objective = function
                if scalar and (objective.input_dimension != 1 or objective.output_dimension != 1):
                    raise ValueError('Scalar mode requires a one-dimensional objective.')
            elif scalar:
                if input_dimension is not None or output_dimension is not None:
                    raise ValueError('Scalar input/output dimensions are fixed at one.')
                objective = LeastSquaresObjective.from_scalar(function, derivative=jacobian,
                    workspace_bytes=options.get('workspace_bytes', 64*1024*1024))
            else:
                n = _PointDimension(x0) if input_dimension is None else input_dimension
                m = (_OutputDimension(function) if output_dimension is None and
                     isinstance(function, RectangularCallback) else output_dimension)
                objective = LeastSquaresObjective(function, n, m, jacobian=jacobian,
                    workspace_bytes=options.get('workspace_bytes', 64*1024*1024))
            self._objective = objective
            # Coordinate coercion is done only by the native solver while the
            # objective and ORIGINAL providers are reserved.
            try:
                self._solver = _OPTIMIZATION_STATES[method](objective, (x0,) if scalar else x0, **options)
            except NativeCallbackError as error:
                if getattr(error, '_cyroot_objective', None) is self._objective:
                    error.result = self._wrap(error.result)
                raise
        finally:
            self._constructing = False

    def _ready(self):
        if getattr(self, '_constructing', True) or self._solver is None:
            raise RuntimeError('Solver facade is not initialized.')
        if self._busy:
            raise RuntimeError('Solver facade is already active.')

    def _wrap(self, raw):
        return OptimizationResult(raw.root[0] if self._scalar else raw.root,
            raw.f_root[0] if self._scalar else raw.f_root, raw.iters,
            (raw.f_calls, raw.jac_calls), raw.precision, raw.error, raw.success,
            None if raw.termination_reason is None else TerminationReason(raw.termination_reason),
            self._method, raw)

    def _invoke(self, operation):
        self._ready()
        self._busy = True
        try:
            try:
                raw = getattr(self._solver, operation)()
            except NativeCallbackError as error:
                if getattr(error, '_cyroot_objective', None) is self._objective:
                    error.result = self._wrap(error.result)
                raise
            return self._wrap(raw)
        finally:
            self._busy = False

    def step(self) -> OptimizationResult:
        return self._invoke('step')

    def run(self) -> OptimizationResult:
        return self._invoke('run')

    @property
    def result(self) -> OptimizationResult:
        self._ready()
        self._busy = True
        try:
            return self._wrap(self._solver.result)
        finally:
            self._busy = False

    @property
    def history(self) -> tuple[OptimizationResult, ...]:
        self._ready()
        self._busy = True
        try:
            return tuple(self._wrap(raw) for raw in self._solver.history)
        finally:
            self._busy = False

    @property
    def native_state(self):
        self._ready()
        return self._solver

    @property
    def objective(self) -> LeastSquaresObjective:
        self._ready()
        return self._objective

    @property
    def done(self) -> bool:
        self._ready()
        return self._solver.done

    @property
    def history_truncated(self) -> bool:
        self._ready()
        return self._solver.history_truncated

    @property
    def memory_bytes(self) -> int:
        self._ready()
        return self._solver.memory_bytes


def find_optimization_root(method: str, function, x0, *, jacobian=None, scalar=False,
                           input_dimension=None, output_dimension=None, **options) -> OptimizationResult:
    return OptimizationSolver(method, function, x0, jacobian=jacobian, scalar=scalar,
        input_dimension=input_dimension, output_dimension=output_dimension, **options).run()


def scalar_levenberg_marquardt(function, derivative, x0, **options) -> OptimizationResult:
    return find_optimization_root('levenberg_marquardt', function, x0, jacobian=derivative,
                                  scalar=True, **options)


def vector_levenberg_marquardt(function, jacobian, x0, *, input_dimension=None,
                               output_dimension=None, **options) -> OptimizationResult:
    return find_optimization_root('levenberg_marquardt', function, x0, jacobian=jacobian,
        input_dimension=input_dimension, output_dimension=output_dimension, **options)


def scalar_bfgs(function, derivative, x0, **options) -> OptimizationResult:
    return find_optimization_root('bfgs', function, x0, jacobian=derivative,
                                  scalar=True, **options)


def vector_bfgs(function, jacobian, x0, *, input_dimension=None,
                output_dimension=None, **options) -> OptimizationResult:
    return find_optimization_root('bfgs', function, x0, jacobian=jacobian,
        input_dimension=input_dimension, output_dimension=output_dimension, **options)


def scalar_lbfgs(function, derivative, x0, **options) -> OptimizationResult:
    return find_optimization_root('lbfgs', function, x0, jacobian=derivative,
                                  scalar=True, **options)


def vector_lbfgs(function, jacobian, x0, *, input_dimension=None,
                 output_dimension=None, **options) -> OptimizationResult:
    return find_optimization_root('lbfgs', function, x0, jacobian=jacobian,
        input_dimension=input_dimension, output_dimension=output_dimension, **options)


def scalar_powell_dogleg(function, derivative, x0, **options) -> OptimizationResult:
    return find_optimization_root('powell_dogleg', function, x0, jacobian=derivative,
                                  scalar=True, **options)


def vector_powell_dogleg(function, jacobian, x0, *, input_dimension=None,
                         output_dimension=None, **options) -> OptimizationResult:
    return find_optimization_root('powell_dogleg', function, x0, jacobian=jacobian,
        input_dimension=input_dimension, output_dimension=output_dimension, **options)


OPTIMIZATION_ROOT_FINDING_METHODS = {
    'levenberg_marquardt': vector_levenberg_marquardt,
    'bfgs': vector_bfgs,
    'lbfgs': vector_lbfgs,
    'powell_dogleg': vector_powell_dogleg,
}

