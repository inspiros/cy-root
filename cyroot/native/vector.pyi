from typing import Any, NamedTuple
from ..status import TerminationReason
from ..return_types import NewtonMethodsReturnType

class VectorNewtonStep(NamedTuple):
    root: tuple[float, ...]
    f_root: tuple[float, ...]
    df_root: tuple[tuple[float, ...], ...]
    iters: int
    f_calls: tuple[int, ...]
    precision: float
    error: float
    termination_reason: TerminationReason | None
    callback_code: int
    callback_stage: str | None
    linear_status: int
    linear_rank: int | None

class WolfeBittnerResult(NamedTuple):
    root: tuple[float, ...]
    f_root: tuple[float, ...]
    active_root: tuple[float, ...]
    active_f_root: tuple[float, ...]
    iters: int
    f_calls: int
    precision: float
    error: float
    termination_reason: TerminationReason | None
    callback_code: int
    callback_stage: str | None
    linear_status: int
    linear_rank: int | None
    affine_error: float
    @property
    def success(self) -> bool: ...
    @property
    def converged(self) -> bool: ...
    @property
    def optimal(self) -> bool: ...

WolfeBittnerStep = WolfeBittnerResult

VectorCallback: Any
VectorNewton: Any
Robinson: Any
TraubSteffensen: Any
WolfeBittner: Any

def vector_newton(function: Any, jacobian: Any, x0: Any, **options: Any) -> NewtonMethodsReturnType: ...
def robinson(function: Any, x0: Any, x1: Any, **options: Any) -> NewtonMethodsReturnType: ...
def traub_steffensen(function: Any, x0: Any, **options: Any) -> NewtonMethodsReturnType: ...
def wolfe_bittner(function: Any, x0s: Any, **options: Any) -> WolfeBittnerResult: ...
