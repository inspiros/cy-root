from typing import Any, NamedTuple, SupportsFloat, SupportsIndex
from numpy.typing import ArrayLike
from .objective import LeastSquaresObjective

class LevenbergMarquardtResult(NamedTuple):
    root: tuple[float, ...]
    f_root: tuple[float, ...]
    error: float
    precision: float
    iters: int
    f_calls: int
    jac_calls: int
    rejected_trials: int
    damping: float
    gradient: float
    success: bool
    termination_reason: str | None
    linear_status: int
    linear_rank: int | None
    callback_code: int
    callback_stage: str | None

class PowellDoglegResult(NamedTuple):
    root: tuple[float, ...]
    f_root: tuple[float, ...]
    error: float
    precision: float
    iters: int
    f_calls: int
    jac_calls: int
    rejected_trials: int
    trust_radius: float
    gradient: float
    success: bool
    termination_reason: str | None
    linear_status: int
    linear_rank: int | None
    callback_code: int
    callback_stage: str | None

class LevenbergMarquardt:
    def __init__(self, objective: LeastSquaresObjective, x0: ArrayLike, *,
                 etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
                 ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
                 gtol: SupportsFloat = ..., max_iter: SupportsIndex = ...,
                 max_evaluations: SupportsIndex = ..., max_trials: SupportsIndex = ...,
                 damping: SupportsFloat = ..., rcond: SupportsFloat = ...,
                 history: SupportsIndex = ..., history_every: SupportsIndex = ...,
                 workspace_bytes: SupportsIndex = ...) -> None: ...
    def step(self) -> LevenbergMarquardtResult: ...
    def run(self) -> LevenbergMarquardtResult: ...
    @property
    def result(self) -> LevenbergMarquardtResult: ...
    @property
    def history(self) -> tuple[LevenbergMarquardtResult, ...]: ...
    @property
    def done(self) -> bool: ...
    @property
    def history_truncated(self) -> bool: ...
    @property
    def memory_bytes(self) -> int: ...

class PowellDogleg(LevenbergMarquardt):
    def __init__(self, objective: LeastSquaresObjective, x0: ArrayLike, *,
                 initial_radius: SupportsFloat = ..., max_radius: SupportsFloat = ...,
                 min_radius: SupportsFloat = ...,
                 etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
                 ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
                 gtol: SupportsFloat = ..., max_iter: SupportsIndex = ...,
                 max_evaluations: SupportsIndex = ..., max_trials: SupportsIndex = ...,
                 rcond: SupportsFloat = ..., history: SupportsIndex = ...,
                 history_every: SupportsIndex = ..., workspace_bytes: SupportsIndex = ...) -> None: ...
    def step(self) -> PowellDoglegResult: ...  # type: ignore[override]
    def run(self) -> PowellDoglegResult: ...  # type: ignore[override]
    @property
    def result(self) -> PowellDoglegResult: ...  # type: ignore[override]
    @property
    def history(self) -> tuple[PowellDoglegResult, ...]: ...  # type: ignore[override]

def levenberg_marquardt(objective: LeastSquaresObjective, x0: ArrayLike, **options: Any) -> LevenbergMarquardtResult: ...
def powell_dogleg(objective: LeastSquaresObjective, x0: ArrayLike, **options: Any) -> PowellDoglegResult: ...
