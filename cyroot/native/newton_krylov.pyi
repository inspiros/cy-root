from typing import Any, NamedTuple, SupportsFloat, SupportsIndex
from collections.abc import Iterator
from numpy.typing import ArrayLike
from ..status import TerminationReason

__all__=['NewtonKrylov','NewtonKrylovResult','newton_krylov']

class NewtonKrylovResult(NamedTuple):
    root: tuple[float, ...]
    f_root: tuple[float, ...]
    iters: int
    f_calls: tuple[int, int]
    precision: float
    error: float
    termination_reason: TerminationReason | None
    linear_iterations: int
    linear_residual: float
    linear_status: int
    linear_stationarity: float
    vjp_calls: int
    step_scale: float
    backtracks: int
    callback_code: int
    callback_stage: str | None
    @property
    def success(self) -> bool: ...
    @property
    def converged(self) -> bool: ...
    @property
    def optimal(self) -> bool: ...

class NewtonKrylov(Iterator[NewtonKrylovResult]):
    def __init__(self, function: Any, x0: ArrayLike, *, jvp: Any = ..., vjp: Any = ...,
                 f_x0: ArrayLike | None = ..., restart: SupportsIndex = ...,
                 max_linear_iter: SupportsIndex = ..., linear_rtol: SupportsFloat = ...,
                 c1: SupportsFloat = ..., contraction: SupportsFloat = ...,
                 max_line_search: SupportsIndex = ..., etol: SupportsFloat = ...,
                 ertol: SupportsFloat = ..., ptol: SupportsFloat = ...,
                 prtol: SupportsFloat = ..., max_iter: SupportsIndex = ...,
                 max_evaluations: SupportsIndex = ..., history: SupportsIndex = ...,
                 history_every: SupportsIndex = ..., history_bytes: SupportsIndex = ...,
                 workspace_bytes: SupportsIndex = ...) -> None: ...
    def step(self) -> NewtonKrylovResult: ...
    def run(self) -> NewtonKrylovResult: ...
    def __iter__(self) -> NewtonKrylov: ...
    def __next__(self) -> NewtonKrylovResult: ...
    @property
    def result(self) -> NewtonKrylovResult: ...
    @property
    def history(self) -> tuple[NewtonKrylovResult, ...]: ...
    @property
    def done(self) -> bool: ...
    @property
    def history_truncated(self) -> bool: ...
    @property
    def workspace_bytes(self) -> int: ...
    @property
    def dimension(self) -> int: ...
    @property
    def restart(self) -> int: ...
    @property
    def output_dimension(self) -> int: ...
    @property
    def linear_solver(self) -> str: ...

def newton_krylov(function: Any, x0: ArrayLike, *, jvp: Any = ..., vjp: Any = ...,
                  **options: Any) -> NewtonKrylovResult: ...
