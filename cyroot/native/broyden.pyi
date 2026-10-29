from collections.abc import Iterator
from typing import Any, NamedTuple, SupportsFloat, SupportsIndex
from numpy.typing import ArrayLike
from ..status import TerminationReason
from ..return_types import NewtonMethodsReturnType

class BroydenResult(NamedTuple):
    root: tuple[float, ...]
    f_root: tuple[float, ...]
    approximation: tuple[tuple[float, ...], ...]
    approximation_kind: str
    approximation_iteration: int
    iters: int
    f_calls: int
    precision: float
    error: float
    termination_reason: TerminationReason | None
    callback_code: int
    callback_stage: str | None
    linear_status: int
    phase: str | None
    approximation_valid: bool
    initialization_calls: int
    @property
    def success(self) -> bool: ...
    @property
    def converged(self) -> bool: ...
    @property
    def optimal(self) -> bool: ...

BroydenStep = BroydenResult

class BroydenGood:
    def __init__(self, function: Any, jacobian0: Any, x0: ArrayLike,
                 f_x0: ArrayLike | None = ..., *,
                 etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
                 ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
                 max_iter: SupportsIndex = ..., history: SupportsIndex = ...,
                 history_every: SupportsIndex = ..., history_bytes: SupportsIndex = ...,
                 workspace_bytes: SupportsIndex = ...) -> None: ...
    def step(self) -> BroydenResult: ...
    def run(self) -> BroydenResult: ...
    @property
    def result(self) -> BroydenResult: ...
    @property
    def history(self) -> tuple[BroydenResult, ...]: ...
    @property
    def done(self) -> bool: ...
    @property
    def history_truncated(self) -> bool: ...
    @property
    def method(self) -> str: ...
    def __iter__(self) -> Iterator[BroydenResult]: ...
    def __next__(self) -> BroydenResult: ...

class BroydenBad(BroydenGood):
    def __init__(self, function: Any, inverse0: Any, x0: ArrayLike,
                 f_x0: ArrayLike | None = ..., *,
                 etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
                 ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
                 max_iter: SupportsIndex = ..., history: SupportsIndex = ...,
                 history_every: SupportsIndex = ..., history_bytes: SupportsIndex = ...,
                 workspace_bytes: SupportsIndex = ...) -> None: ...
    @classmethod
    def from_jacobian(cls, function: Any, jacobian0: Any, x0: ArrayLike,
                      f_x0: ArrayLike | None = ..., *,
                      etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
                      ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
                      max_iter: SupportsIndex = ..., history: SupportsIndex = ...,
                      history_every: SupportsIndex = ..., history_bytes: SupportsIndex = ...,
                      workspace_bytes: SupportsIndex = ...) -> BroydenBad: ...

class Klement(BroydenGood): ...

class Barnes(BroydenGood):
    @property
    def secant_history_size(self) -> int: ...

def broyden_good(function: Any, jacobian0: Any, x0: ArrayLike, **options: Any) -> BroydenResult: ...
def broyden_bad(function: Any, inverse0: Any, x0: ArrayLike, **options: Any) -> BroydenResult: ...
def barnes(function: Any, jacobian0: Any, x0: ArrayLike, **options: Any) -> BroydenResult: ...
def klement(function: Any, jacobian0: Any, x0: ArrayLike, **options: Any) -> BroydenResult: ...
