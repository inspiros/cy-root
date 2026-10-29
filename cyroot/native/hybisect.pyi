from collections.abc import Iterator
from typing import Any, NamedTuple, SupportsFloat, SupportsIndex
from ..status import TerminationReason
from ..return_types import SplittingBracketingMethodsReturnType

class HybisectStep(NamedTuple):
    bracket: tuple[float, float]
    root: float
    f_root: float
    roots_found: int
    split_iters: int
    iters: int
    f_calls: tuple[int, int]
    pending: int
    event: str
    termination_reason: TerminationReason | None

class IntervalCallback:
    def __init__(self, function: Any) -> None: ...
    @classmethod
    def from_capsule(cls, capsule: Any, keepalive: Any = ...) -> IntervalCallback: ...
    @property
    def native(self) -> bool: ...

class Hybisect:
    def __init__(self, function: Any, interval_function: Any, a: SupportsFloat, b: SupportsFloat, *,
                 etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
                 ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
                 max_iter: SupportsIndex = ..., max_split_iter: SupportsIndex = ...,
                 max_pending: SupportsIndex = ..., max_roots: SupportsIndex = ...,
                 workspace_bytes: SupportsIndex = ..., history: SupportsIndex = ...,
                 history_every: SupportsIndex = ..., history_bytes: SupportsIndex = ...) -> None: ...
    def step(self) -> HybisectStep: ...
    def run(self) -> SplittingBracketingMethodsReturnType: ...
    @property
    def result(self) -> SplittingBracketingMethodsReturnType: ...
    @property
    def done(self) -> bool: ...
    @property
    def history(self) -> tuple[HybisectStep, ...]: ...
    @property
    def history_truncated(self) -> bool: ...
    @property
    def workspace_exhausted(self) -> bool: ...
    def __iter__(self) -> Iterator[HybisectStep]: ...
    def __next__(self) -> HybisectStep: ...

def hybisect(function: Any, interval_function: Any, a: SupportsFloat, b: SupportsFloat,
             **options: Any) -> SplittingBracketingMethodsReturnType: ...
