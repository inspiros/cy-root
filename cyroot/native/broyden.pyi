from typing import Any, NamedTuple
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

BroydenGood: Any
BroydenBad: Any
Barnes: Any
Klement: Any

def broyden_good(function: Any, jacobian0: Any, x0: Any, **options: Any) -> BroydenResult: ...
def broyden_bad(function: Any, inverse0: Any, x0: Any, **options: Any) -> BroydenResult: ...
def barnes(function: Any, jacobian0: Any, x0: Any, **options: Any) -> BroydenResult: ...
def klement(function: Any, jacobian0: Any, x0: Any, **options: Any) -> BroydenResult: ...
