from typing import Any, NamedTuple, SupportsFloat, SupportsIndex
from numpy.typing import ArrayLike
from .objective import LeastSquaresObjective

class BFGSResult(NamedTuple):
    root: tuple[float, ...]
    f_root: tuple[float, ...]
    error: float
    precision: float
    value: float
    gradient: float | None
    objective_scale: float
    step_size: float
    iters: int
    f_calls: int
    jac_calls: int
    rejected_trials: int
    skipped_updates: int
    restarts: int
    success: bool
    termination_reason: str | None
    callback_code: int
    callback_stage: str | None

class BFGS:
    def __init__(self, objective: LeastSquaresObjective, x0: ArrayLike, *,
                 etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
                 ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
                 gtol: SupportsFloat = ..., max_iter: SupportsIndex = ...,
                 max_evaluations: SupportsIndex = ..., max_line_search: SupportsIndex = ...,
                 c1: SupportsFloat = ..., c2: SupportsFloat = ...,
                 history: SupportsIndex = ..., history_every: SupportsIndex = ...,
                 workspace_bytes: SupportsIndex = ...) -> None: ...
    def step(self) -> BFGSResult: ...
    def run(self) -> BFGSResult: ...
    @property
    def result(self) -> BFGSResult: ...
    @property
    def history(self) -> tuple[BFGSResult, ...]: ...
    @property
    def done(self) -> bool: ...
    @property
    def history_truncated(self) -> bool: ...
    @property
    def memory_bytes(self) -> int: ...

class LBFGS(BFGS):
    def __init__(self, objective: LeastSquaresObjective, x0: ArrayLike, *,
                 corrections: SupportsIndex = ...,
                 etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
                 ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
                 gtol: SupportsFloat = ..., max_iter: SupportsIndex = ...,
                 max_evaluations: SupportsIndex = ..., max_line_search: SupportsIndex = ...,
                 c1: SupportsFloat = ..., c2: SupportsFloat = ...,
                 history: SupportsIndex = ..., history_every: SupportsIndex = ...,
                 workspace_bytes: SupportsIndex = ...) -> None: ...

def bfgs(objective: LeastSquaresObjective, x0: ArrayLike, **options: Any) -> BFGSResult: ...
def lbfgs(objective: LeastSquaresObjective, x0: ArrayLike, **options: Any) -> BFGSResult: ...
