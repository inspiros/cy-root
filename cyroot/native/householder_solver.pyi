from collections.abc import Iterator, Sequence
from typing import Any, NamedTuple, SupportsFloat, SupportsIndex
from ..status import TerminationReason
from ..return_types import NewtonMethodsReturnType

class HouseholderStep(NamedTuple):
    root: float
    f_root: float
    df_root: tuple[float, ...]
    iters: int
    f_calls: tuple[int, ...]
    precision: float
    error: float
    termination_reason: TerminationReason | None
    callback_code: int
    callback_stage: str | None
    callback_order: int | None

class Householder:
    def __init__(self, function: Any, derivatives: list[Any] | tuple[Any, ...] | None, x0: SupportsFloat,
                 f_x0: SupportsFloat | None = ...,
                 dfs_x0: Sequence[SupportsFloat | None] | None = ..., *,
                 order: SupportsIndex | None = ..., step: SupportsFloat | None = ...,
                 etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
                 ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
                 max_iter: SupportsIndex = ..., history: SupportsIndex = ...,
                 history_every: SupportsIndex = ..., history_bytes: SupportsIndex = ...,
                 workspace_bytes: SupportsIndex = ..., max_nodes: SupportsIndex = ...) -> None: ...
    def step(self) -> HouseholderStep: ...
    def run(self) -> NewtonMethodsReturnType: ...
    @property
    def result(self) -> NewtonMethodsReturnType: ...
    @property
    def history(self) -> tuple[HouseholderStep, ...]: ...
    @property
    def done(self) -> bool: ...
    @property
    def order(self) -> int: ...
    @property
    def history_truncated(self) -> bool: ...
    @property
    def callback_code(self) -> int: ...
    @property
    def callback_stage(self) -> str | None: ...
    @property
    def callback_order(self) -> int | None: ...
    def __iter__(self) -> Iterator[HouseholderStep]: ...
    def __next__(self) -> HouseholderStep: ...

def householder(function: Any, derivatives: list[Any] | tuple[Any, ...] | None, x0: SupportsFloat,
                **options: Any) -> NewtonMethodsReturnType: ...
