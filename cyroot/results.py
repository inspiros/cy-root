"""Copied scalar/vector state for inspecting native solver iterations."""
from collections.abc import Iterable
from numbers import Complex, Real
from operator import index
from typing import Any, NamedTuple, Protocol, TypeAlias

from .return_types import MultiRootsReturnType
from .status import TerminationReason

__all__ = ['RootValue', 'SnapshotSource', 'RootSnapshot', 'as_root_snapshot']

RootValue: TypeAlias = float | complex | tuple[float, ...]


class SnapshotSource(Protocol):
    """The read-only fields shared by single-root results and native steps."""
    @property
    def root(self) -> Any: ...
    @property
    def f_root(self) -> Any: ...
    @property
    def iters(self) -> int: ...
    @property
    def f_calls(self) -> int | tuple[int, ...]: ...
    @property
    def precision(self) -> float: ...
    @property
    def error(self) -> float: ...
    @property
    def termination_reason(self) -> str | None: ...


class RootSnapshot(NamedTuple):
    """Immutable common values, with the original state in diagnostics.

    Counts are a tuple: residual first, then derivatives in the source's order.
    Native optimizer Jacobian counts are included after residual counts.
    None means a running or unspecified termination reason. Bracket tolerance
    can establish success without residual convergence; inspect the reason.
    """
    root: RootValue
    f_root: RootValue
    iters: int
    f_calls: tuple[int, ...]
    precision: float
    error: float
    success: bool
    termination_reason: TerminationReason | None
    diagnostics: object

    @property
    def residual_converged(self) -> bool:
        return self.termination_reason == TerminationReason.RESIDUAL_TOLERANCE


def _value(value: Any) -> RootValue:
    if isinstance(value, (str, bytes)):
        raise TypeError('Root values must be numeric scalars or vectors.')
    if isinstance(value, Complex) and not isinstance(value, Real):
        return complex(value)
    try:
        values = iter(value)
    except TypeError:
        return float(value)
    return tuple(float(item) for item in values)


def as_root_snapshot(source: SnapshotSource) -> RootSnapshot:
    """Copy one result/step; use map(as_root_snapshot, solver.history).

    This function neither runs a solver nor evaluates its callbacks. It retains
    the original diagnostic object, while copying scalar/vector values. Interval
    search events and multi-root results retain their separate search interface.
    """
    if isinstance(source, RootSnapshot):
        return source
    if isinstance(source, MultiRootsReturnType):
        raise TypeError('Multi-root searches use their per-root and search fields.')
    try:
        precision, error = source.precision, source.error
    except AttributeError:
        raise TypeError('Expected a single-root result or step, not a search event.') from None
    raw_reason = source.termination_reason
    reason = None if raw_reason is None else TerminationReason(raw_reason)
    if reason == TerminationReason.UNSPECIFIED:
        reason = None
    counts = source.f_calls
    if isinstance(counts, Iterable):
        calls = tuple(index(count) for count in counts)
    else:
        calls = (index(counts),)
        jac_calls = getattr(source, 'jac_calls', None)
        if jac_calls is not None:
            calls += (index(jac_calls),)
    success = getattr(source, 'success', None)
    if success is None:
        success = reason in (TerminationReason.RESIDUAL_TOLERANCE, TerminationReason.BRACKET_TOLERANCE)
    return RootSnapshot(_value(source.root), _value(source.f_root), index(source.iters),
                        calls, float(precision), float(error), bool(success), reason, source)
