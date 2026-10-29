from collections.abc import Iterator
from typing import Any, NamedTuple, SupportsFloat, SupportsIndex
from ..status import TerminationReason
from ..return_types import RootReturnType

class SecantStep(NamedTuple):
    root: float
    f_root: float
    previous_root: float
    previous_f_root: float
    iters: int
    f_calls: int
    precision: float
    error: float
    termination_reason: TerminationReason | None
    callback_code: int

class Secant:
    def __init__(self, function: Any, x0: SupportsFloat, x1: SupportsFloat,
                 f_x0: SupportsFloat | None = ..., f_x1: SupportsFloat | None = ...,
                 etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
                 ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
                 max_iter: SupportsIndex = ..., history: SupportsIndex = ...,
                 history_every: SupportsIndex = ..., history_bytes: SupportsIndex = ...) -> None: ...
    def step(self) -> SecantStep: ...
    def run(self) -> RootReturnType: ...
    @property
    def result(self) -> RootReturnType: ...
    @property
    def history(self) -> tuple[SecantStep, ...]: ...
    @property
    def done(self) -> bool: ...
    @property
    def history_truncated(self) -> bool: ...
    @property
    def callback_code(self) -> int: ...
    @property
    def callback_stage(self) -> str | None: ...
    def __iter__(self) -> Iterator[SecantStep]: ...
    def __next__(self) -> SecantStep: ...

def secant(function: Any, x0: SupportsFloat, x1: SupportsFloat, **options: Any) -> RootReturnType: ...
